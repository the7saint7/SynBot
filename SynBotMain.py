import os
import io
import json
import base64
import aiohttp
import asyncio
import discord
import requests
from PIL import Image
from dotenv import load_dotenv
from LORA_Helper import LORA_List
from discord.ext import tasks, commands
from charactersList import charactersLORA
from openPoses import getPose, getLewdPose, getImageAtPath, getBase64FromImage, getImageFormBase64

load_dotenv()

class SynBotManager(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Our request queue
        self.queue = asyncio.Queue()

    async def setup_hook(self) -> None:
        # start the task to run in the background
        self.my_background_task.start()

    @tasks.loop(seconds=5)  # task runs every 5 seconds
    async def my_background_task(self):
        print("dequeing...")
        task = await self.queue.get()
        if isinstance(task, SynBotPrompt):
            await task.generateImage()
        else:
            await task
        self.queue.task_done() 

    @my_background_task.before_loop
    async def before_my_task(self):
        await self.wait_until_ready()  # wait until the bot logs in

class SynBotPrompt:
    def __init__(self, context, outputChanel, type=None):
        self.ctx = context
        self.outputChanel = outputChanel

        # Default parameters
        self.type = type # txt2img, img2img, inpaint, outfits 
        self.isValid = True
        self.hirez = False
        self.hirezValue = 1.0
        self.seedToUse = -1
        self.batchCount = 1
        self.poseNumber = None
        self.lewdPoseNumber = None
        self.removeBG = False
        self.format = "640x360" 
        self.userNegative = "NEGATIVE" # What the user sent originaly
        self.fixedNegative = "" # after replacing the tags with the real prompts
        self.userPrompt = "" # What the user sent originaly
        self.fixedPrompt = "" # after replacing the tags with the real prompts
        self.URL = ""
        self.denoise = 0.5
        self.errorMsg = None

        # img2img
        self.img2imgImage = None
        self.img2img_width = 640
        self.img2img_height = 360

        # inpaint
        self.inpaintBaseImage = None
        self.inpaintMaskImage = None

        # outfits
        self.outfitsCharacter = None
        self.outfitsPose = None
        self.outfitsName = None

        # birth
        self.birthPoses = []
        self.birthBaseImage = None

        # # The message that was sent
        message = str(self.ctx.message.content)

        # Remove all possible prefixes
        prefixes = ["!Syn-txt2img", "!Syn-img2img", "!Syn-inpaint", "!Syn-outfits", "!Syn2-txt2img", "!Syn2-img2img", "!Syn2-inpaint", "!Syn2-outfits", "!Syn-birth", "!Syn2-birth"]
        for prefix in prefixes:
            message = message.removeprefix(prefix)

        # Try to load the JSON data, return if its invalid
        try:
            jsonData = json.loads(message.strip())
        except ValueError as e:
            self.errorMsg = f"{self.ctx.author.mention} invalid json: {e}"
            print(f"invalid json: {e} -> {message}")
            self.isValid = False
            return

        # Required parameters
        if "prompt" in jsonData:
            self.userPrompt = jsonData["prompt"]
        else:
            self.errorMsg = f"{self.ctx.author.mention} Missing **'prompt'** parameter"
            self.isValid = False
            return

        ###### TXT2IMG specific init
        if self.type == "txt2img":
            if "format" in jsonData:
                self.formatStr = jsonData["format"]
            else:
                self.errorMsg = f"{self.ctx.author.mention} Missing **'format'** parameter"
                self.isValid = False
                return
        
            if self.formatStr != "landscape" and self.formatStr != "portrait":
                self.errorMsg = f"{self.ctx.author.mention} Non supported format: '{self.formatStr}'"
                print("non conform format: " + self.formatStr)
                self.isValid = False
                return
            
            self.format = "640x360" if self.formatStr.strip() == "landscape" else "360x640"
        ###### END TXT2IMG specific init

        ###### IMG2IMG specific init
        if self.type == "img2img":
            hasAttachments = len(self.ctx.message.attachments) >= 1
            if not hasAttachments:
                self.errorMsg = f"{self.ctx.author.mention} Missing a file attachment in IMG2IMG command"
                self.isValid = False
                return
            else:

                if "denoise" in jsonData:
                    self.denoise = jsonData["denoise"]
                else:
                    self.errorMsg = f"{self.ctx.author.mention} Missing **'denoise'** parameter"
                    self.isValid = False
                    return

                ###################### START lOADING USER SUBMITTED IMAGE
                print("Loading sent image as attachments...")
                self.img2imgImage = self.encode_discord_image(self.ctx.message.attachments[0].url) # Base?
                print("Attachments loaded in memory.")
                if self.img2imgImage == None:
                    self.errorMsg = "Could not read sent image. Request stopped."
                    self.isValid = False
                    return
                ###################### END LOADING USER SUBMITTED IMAGE


        ###### END IMG2IMG specific init


        ###### INPAINT specific init
        if self.type == "inpaint":
            hasAttachments = len(self.ctx.message.attachments) >= 2
            if not hasAttachments:
                self.errorMsg = f"{self.ctx.author.mention} Missing 2 file attachment in INPAINT command"
                self.isValid = False
                return
            else:

                if "denoise" in jsonData:
                    self.denoise = jsonData["denoise"]
                else:
                    self.errorMsg = f"{self.ctx.author.mention} Missing **'denoise'** parameter"
                    self.isValid = False
                    return

                ###################### START lOADING USER SUBMITTED BASE AND MASK IMAGE
                print("Loading sent images as attachments...")
                self.inpaintBaseImage = self.encode_discord_image(self.ctx.message.attachments[0].url) # Base?
                self.inpaintMaskImage = self.encode_discord_image(self.ctx.message.attachments[1].url) # Mask?
                print("Attachments loaded in memory.")
                if self.inpaintBaseImage == None or self.inpaintMaskImage == None:
                    self.errorMsg = "Could not read sent images. Request stopped."
                    self.isValid = False
                    return
                ###################### END LOADING USER SUBMITTED BASE AND MASK IMAGE


        ###### END INPAINT specific init

        ###### OUTFITS specific init
        if self.type == "outfits":
            
            # Default removeBG True if not specified in prompt parameters (will be overwritten a few lines bellow if specified in parameters)
            self.removeBG = True

            # denoise is optional, no returning an error
            if "denoise" in jsonData:
                self.denoise = jsonData["denoise"]
            else:
                self.denoise = .90

            # Character name
            self.outfitsCharacter = jsonData['character'] if "character" in jsonData else None
            if self.outfitsCharacter == None:
                self.errorMsg = f"{self.ctx.author.mention} Missing **'character'** parameter"
                self.isValid = False
                return
            
            # Character Pose
            self.outfitsPose = jsonData["pose"] if "pose" in jsonData else None
            if self.outfitsPose == None:
                self.errorMsg = f"{self.ctx.author.mention} Missing **'pose'** parameter"
                self.isValid = False
                return

            # Outfit Name; casual, nude, uniform, ect
            self.outfitsName = jsonData["outfit"] if "outfit" in jsonData else None
            if self.outfitsName == None:
                self.errorMsg = f"{self.ctx.author.mention} Missing **'outfit'** parameter"
                self.isValid = False
                return

            # might need to use custom mask
            mask = "mask"
            if self.outfitsCharacter == "allison" and self.outfitsName == "gym":
                mask = "gym_mask"

            # Using the INPAINT base and mask image, as we're basically doing inpaint
            ###################### START DEFAULT BASE AND MASK IMAGE
            baseImagePath = f"./sprites/{self.outfitsCharacter}/{self.outfitsPose}_{self.outfitsName}.png"
            if not os.path.exists(baseImagePath):
                self.errorMsg = f"{self.ctx.author.mention} -> {baseImagePath} does not exist."
                self.isValid = False
                return
            self.inpaintBaseImage = getImageAtPath(baseImagePath)
            
            maskImagePath = f"./sprites/{self.outfitsCharacter}/{self.outfitsPose}_{mask}.png"
            if not os.path.exists(maskImagePath):
                self.errorMsg = f"{self.ctx.author.mention} -> {maskImagePath} does not exist."
                self.isValid = False
                return
            self.inpaintMaskImage = getImageAtPath(maskImagePath)
            ###################### END DEFAULT BASE AND MASK IMAGE

        ###### END OUTFITS specific init

        ###### BIRTH specific init
        if self.type == "birth":
            
            # Default removeBG True if not specified in prompt parameters (will be overwritten a few lines bellow if specified in parameters)
            self.removeBG = True

            # Character Pose
            self.birthPoses = jsonData["birthPoses"] if "birthPoses" in jsonData else None
            if self.birthPoses == None:
                self.errorMsg = f"{self.ctx.author.mention} Missing **'birthPoses'** parameter"
                self.isValid = False
                return
            if len(self.birthPoses) > 2:
                self.errorMsg = f"{self.ctx.author.mention} Too many poses in **'birthPoses'** parameter. Maximum of 2 poses allowed."
                self.isValid = False
                return
            
            # Loop the poses and get their bytes? Then merge them into a single image, side by side
            poseImages = []
            for pose in self.birthPoses:
                posePath = f"./poses/portrait_cowboy_shot/{pose}.png"
                if not os.path.exists(posePath):
                    self.errorMsg = f"{self.ctx.author.mention} -> {posePath} does not exist."
                    self.isValid = False
                    return
                poseImages.append(Image.open(posePath))

            # Concatenate the images
            widths, heights = zip(*(i.size for i in poseImages))

            total_width = sum(widths)
            max_height = max(heights)

            new_im = Image.new('RGB', (total_width, max_height))

            x_offset = 0
            for im in poseImages:
                new_im.paste(im, (x_offset,0))
                x_offset += im.size[0]
            
            self.birthBaseImage = new_im


        ###### END BIRTH specific init


        # Optional parameters
        if "negative" in jsonData: self.userNegative = jsonData["negative"]
        if "hirez" in jsonData: self.hirez = jsonData["hirez"] == "true"
        if "batch" in jsonData: self.batchCount = jsonData["batch"]
        if "seed" in jsonData: self.seedToUse = jsonData["seed"]
        if "pose" in jsonData: self.poseNumber = jsonData["pose"]
        if "lewdPose" in jsonData: self.lewdPoseNumber = jsonData["lewdPose"]
        if "removeBG" in jsonData: self.removeBG = jsonData["removeBG"] == "true"
        if "lewdPose" in jsonData: self.lewdPoseNumber = jsonData["lewdPose"]

        # Reset batchCount if hirez
        if self.hirez and self.batchCount != 1:
            self.batchCount = 1

        ############### Replace prompt tags
        self.fixedPrompt = self.userPrompt
        self.fixedNegative = self.userNegative

        for key in charactersLORA.keys():
            if key in self.fixedPrompt:
                print("Found: " + key)
                self.fixedPrompt = self.fixedPrompt.replace(key, charactersLORA[key])

        # Same for LORA Helpers
        for key in LORA_List.keys():
            if key in self.fixedPrompt:
                print("Found: " + key + " in prompt")
                self.fixedPrompt = self.fixedPrompt.replace(key, LORA_List[key])
            if key in self.userNegative:
                print("Found: " + key + " in negative")
                self.fixedNegative = self.fixedNegative.replace(key, LORA_List[key])
        ############### END Replace prompt tags

    # Process an image URL and return a base64 encoded string
    def encode_discord_image(self, image_url):
        try:
            response = requests.get(image_url)
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")

            # For img2img and maybe inpaint
            width, height = image.size
            self.img2img_width = width
            self.img2img_height = height

            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Error in encode_discord_image: {e}")
            return None

    async def removeBackground(self, discordFile, URL, ctx):

        # for discordFile in discordFiles:
        payload = {
            "input_image": discordFile,
            "model": "isnet-anime"
        }

        # Sending API call request
        print(f"Sending request to '{URL}' ...")
        async with aiohttp.ClientSession(loop=ctx.bot.loop) as session:
            async with session.post(url=f'{URL}/rembg', json=payload) as response:
                print("Request returned: " + str(response.status))
                if response.status == 200:
                    r = await response.json()
                    return r['image']
    
    async def generateImage(self):
        
        ######################### START PAYLOAD BUILDING #####################################
        #########################        TXT2IMG         #####################################
        if self.type == "txt2img":
            
            # Where do we end the request?
            apiPath = "/sdapi/v1/txt2img"

            payload = {
                "prompt": self.fixedPrompt,
                "negative_prompt": self.fixedNegative,
                "sampler_name": "DPM++ 2M Karras",
                "batch_size": self.batchCount,
                "steps": 35,
                "cfg_scale": 7,
                "width": int(self.format.split("x")[0]),
                "height": int(self.format.split("x")[1]),
                "restore_faces": False,
                "seed": self.seedToUse
            }

            if self.hirez:
                payload["denoising_strength"] = 0.5
                payload["enable_hr"] = True
                payload["hr_upscaler"] = "4x-UltraSharp"
                #payload["hr_resize_x"] = int(format.split("x")[0]) * HIREZ_SCALE
                #payload["hr_resize_y"] = int(format.split("x")[1]) * HIREZ_SCALE
                payload["hr_scale"] = self.hirezValue
                payload["hr_sampler_name"] = "DPM++ 2M Karras"
                payload["hr_second_pass_steps"] = 20
            
            poseImage = None
            if self.poseNumber != None:
                # Pick a pose according toe format and "shot"
                pose_format = "landscape" if int(self.format.split("x")[0]) > int(self.format.split("x")[1]) else "portrait"
                pose_shot = "full_body" if "full_body" in self.userPrompt else "cowboy_shot"
                poseImage = getPose(pose_format, pose_shot, self.poseNumber)

            if self.lewdPoseNumber != None:
                # Pick a pose according toe format and "shot"
                poseImage = getLewdPose(self.lewdPoseNumber)

            if poseImage != None:
                payload["alwayson_scripts"] = {
                    "controlnet": {
                        "args": [
                            {
                                "input_image": poseImage,
                                "model": "control_v11p_sd15_openpose [cab727d4]",
                                "weight": 1,
                                # "width": 512,
                                # "height": 768,
                                "pixel_perfect": True
                            }
                        ]
                    }
                }

        #########################         IMG2IMG        #####################################
        elif self.type == "img2img":

            # Where do we end the request?
            apiPath = "/sdapi/v1/img2img"

            payload = {
                "init_images": [ self.img2imgImage ], 
                "denoising_strength": self.denoise, 
                "image_cfg_scale": 7, 
                "sampler_name": "DPM++ 2M Karras", 
                "batch_size": self.batchCount, 
                "steps": 30,
                "seed": self.seedToUse, 
                "cfg_scale": 7, 
                "width": self.img2img_width, "height": self.img2img_height, 
                "prompt": "masterpiece, best_quality, extremely detailed, intricate, high_details, sharp_focus , best_anatomy, hires, (colorful), beautiful, 4k, magical, adorable, (extraordinary:0.6), (((simple_background))), (((white_background))), multiple_views, reference_sheet, " + self.fixedPrompt, 
                "negative_prompt": "paintings, sketches, fingers, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, (outdoor:1.6), backlight,(ugly:1.331), (duplicate:1.331), (morbid:1.21), (mutilated:1.21), (tranny:1.331), mutated hands, (poorly drawn hands:1.5), blurry, (bad anatomy:1.21), (bad proportions:1.331), extra limbs, (disfigured:1.331), lowers, bad hands, missing fingers, extra digit, " + self.fixedNegative, 
                "sampler_index": "DPM++ 2M Karras"
            }

            
        #########################        INPAINT       #####################################
        elif self.type == "inpaint":

            # Where do we end the request?
            apiPath = "/sdapi/v1/img2img"

            # This is how rescaled the inpainting will be. I should make this another parameter TODO
            width = 728
            height = 728
            if self.hirez:
                width = 856
                height = 856

            payload = {
                "init_images": [ self.inpaintBaseImage ], 
                "mask": self.inpaintMaskImage, 
                "denoising_strength": self.denoise, 
                "image_cfg_scale": 7, 
                "mask_blur": 10, 
                "inpaint_full_res": True,                   #choices=["Whole picture", "Only masked"]
                "inpaint_full_res_padding": 32, 
                "inpainting_mask_invert": 0,                #choices=['Inpaint masked', 'Inpaint not masked']
                "sampler_name": "DPM++ 2M Karras", 
                "batch_size": self.batchCount, 
                "steps": 30,
                "seed": self.seedToUse, 
                "cfg_scale": 7, 
                "width": width, "height": height, 
                "prompt": "masterpiece, best_quality, extremely detailed, intricate, high_details, sharp_focus , best_anatomy, hires, (colorful), beautiful, 4k, magical, adorable, (extraordinary:0.6), (((simple_background))), (((white_background))), multiple_views, reference_sheet, " + self.fixedPrompt, 
                "negative_prompt": "paintings, sketches, fingers, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, (outdoor:1.6), backlight,(ugly:1.331), (duplicate:1.331), (morbid:1.21), (mutilated:1.21), (tranny:1.331), mutated hands, (poorly drawn hands:1.5), blurry, (bad anatomy:1.21), (bad proportions:1.331), extra limbs, (disfigured:1.331), lowers, bad hands, missing fingers, extra digit, " + self.fixedNegative, 
                "sampler_index": "DPM++ 2M Karras"
            }

            # Add controlNet OpenPose
            payload["alwayson_scripts"] = {
                "controlnet": {
                    "args": [
                        {
                            "enabled": True,
                            "input_image": self.inpaintBaseImage,
                            "module": "openpose",
                            "model": "control_v11p_sd15_openpose [cab727d4]",
                            "weight": .75,  # Apply pose on 75% of the steps
                            "pixel_perfect": True
                        },
                        {
                            "enabled": True,
                            "input_image": self.inpaintBaseImage,
                            "module": "depth_midas",
                            "model": "control_v11f1p_sd15_depth [cfd03158]",
                            "weight": 0.5, # Apply depth only 50% of the steps
                            "guidance": 1.0,
                            "guidance_start": 0.0,
                            "guidance_end": 0.5,
                            "pixel_perfect": True
                        }
                    ]
                }
            }    


        #########################        OUTFITS       #####################################
        elif self.type == "outfits":

            # Where do we end the request?
            apiPath = "/sdapi/v1/img2img"

            # This is how rescaled the inpainting will be.
            width = 728
            height = 728
            if self.hirez:
                width = 856
                height = 856

            # API Payload
            payload = {
                "init_images": [ self.inpaintBaseImage ], 
                "mask": self.inpaintMaskImage, 
                "denoising_strength": self.denoise, 
                "image_cfg_scale": 7, 
                "mask_blur": 10, 
                "inpaint_full_res": True,                   #choices=["Whole picture", "Only masked"]
                "inpaint_full_res_padding": 32, 
                "inpainting_mask_invert": 0,                #choices=['Inpaint masked', 'Inpaint not masked']
                "sampler_name": "DPM++ 2M Karras", 
                "batch_size": self.batchCount, 
                "steps": 30,
                "seed": self.seedToUse, 
                "cfg_scale": 7, 
                "width": width, "height": height, 
                "prompt": "masterpiece, best_quality, extremely detailed, intricate, high_details, sharp_focus , best_anatomy, hires, (colorful), beautiful, 4k, magical, adorable, (extraordinary:0.6), (((simple_background))), (((white_background))), multiple_views, reference_sheet, " + self.fixedPrompt, 
                "negative_prompt": "paintings, sketches, fingers, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, (outdoor:1.6), backlight,(ugly:1.331), (duplicate:1.331), (morbid:1.21), (mutilated:1.21), (tranny:1.331), mutated hands, (poorly drawn hands:1.5), blurry, (bad anatomy:1.21), (bad proportions:1.331), extra limbs, (disfigured:1.331), lowers, bad hands, missing fingers, extra digit, " + self.fixedNegative, 
                "sampler_index": "DPM++ 2M Karras"
            }

            # Add controlNet OpenPose
            payload["alwayson_scripts"] = {
                "controlnet": {
                    "args": [
                        {
                            "enabled": True,
                            "input_image": self.inpaintBaseImage,
                            "module": "openpose",
                            "model": "control_v11p_sd15_openpose [cab727d4]",
                            "weight": .75,  # Apply pose on 75% of the steps
                            "pixel_perfect": True
                        },
                        {
                            "enabled": True,
                            "input_image": self.inpaintBaseImage,
                            "module": "depth_midas",
                            "model": "control_v11f1p_sd15_depth [cfd03158]",
                            "weight": 0.5, # Apply depth only 50% of the steps
                            "guidance": 1.0,
                            "guidance_start": 0.0,
                            "guidance_end": 0.5,
                            "pixel_perfect": True
                        }
                    ]
                }
            }

        #########################         BIRTH        #####################################
        elif self.type == "birth":
            
            # Where do we end the request?
            apiPath = "/sdapi/v1/txt2img"

            payload = {
                "prompt": "((((multiple_views)))), ((solo)), ((solo_focus)), ((from_above)), ((reference sheet)), (((simple_background))), (((white_background))), ((personification)), <lora:multiple views:1>, <lora:Candysprite style:.2>, <lora:Koku_V1.0a:.2>, flat, " + self.fixedPrompt,
                "negative_prompt": self.fixedNegative + ", shadows",
                "sampler_name": "DPM++ 2M Karras",
                "batch_size": self.batchCount,
                "steps": 35,
                "cfg_scale": 7,
                "width": self.birthBaseImage.width / 2,
                "height": self.birthBaseImage.height / 2,
                "restore_faces": False,
                "seed": self.seedToUse
            }

            # Hirez by 2 to recover from our small size
            payload["denoising_strength"] = 0.45
            payload["enable_hr"] = True
            payload["hr_upscaler"] = "4x-UltraSharp"
            payload["hr_scale"] = 2.0                         
            payload["hr_sampler_name"] = "DPM++ 2M Karras"
            payload["hr_second_pass_steps"] = 20
            
            payload["alwayson_scripts"] = {
                "controlnet": {
                    "args": [
                        {
                            "input_image": getBase64FromImage(self.birthBaseImage),
                            "model": "control_v11p_sd15_openpose [cab727d4]",
                            "weight": 1,
                            "width": self.birthBaseImage.width / 2,
                            "height": self.birthBaseImage.height / 2,
                            "pixel_perfect": True
                        }
                    ]
                }
            }
        ######################### END PAYLOAD BUILDING #####################################
        
        # Sending API call request
        print(f"Sending request to {self.URL}{apiPath} ...")
        async with aiohttp.ClientSession(loop=self.ctx.bot.loop) as session:
            async with session.post(url=f'{self.URL}{apiPath}', json=payload)as response:
                print("Request returned: " + str(response.status))
                if response.status == 200:
                    r = await response.json()
                    # Extract the seed that was used to generate the image
                    info = r["info"]
                    infoJson = json.loads(info)
                    responseSeedUsed = infoJson["seed"]

                    # looping response to get actual image
                    discordFiles = []
                    for i in r['images']:

                        # Skip/Remove ControlNet images from output
                        if self.type == "inpaint" or self.type == "outfits" or self.type == "birth":
                            batchSize = payload["batch_size"]
                            if r["images"].index(i) >= batchSize: #Last 2 images are always ControlNet, Depth or OpenPose
                                continue # loop to next image

                        # Remove background
                        if self.removeBG:
                            if self.type != "birth" or not self.hirez: # removing the BG before superHirez is a bad idea
                                i = await self.removeBackground(i, self.URL, self.ctx) if self.removeBG else i


                        # Image is in base64, convert it to a discord.File
                        bytes = io.BytesIO(base64.b64decode(i.split(",",1)[0]))
                        bytes.seek(0)
                        discordFile = discord.File(bytes, filename="{seed}-{ctx.message.author}.png")
                        discordFiles.append(discordFile)


                    # get available tags
                    tags = self.outputChanel.available_tags
                    # Prepare the tag to give to the new thread
                    forumTag = None
                    for tag in tags:
                        if tag.name.lower() == self.type.lower():
                            forumTag = tag
                            break

                    
                    # if the type is "birth" and "hirez", send do a super-hirez on the image
                    if self.type == "birth" and self.hirez:
                        await self.superHirez(r['images'][0])
                        return



                    await self.outputChanel.create_thread(
                        name=f"{self.type.upper()} by {self.ctx.message.author.display_name}", 
                        content=f"{self.ctx.author.mention} generated this image with prompt:\n```{self.ctx.message.content}``` and seed: {responseSeedUsed}", 
                        files=discordFiles,
                        applied_tags=[forumTag]
                    )
                
                else:
                    await self.ctx.send(f"{self.ctx.author.mention} -> API server returned an unknown error. Try again?")
                    print(response)
        


    # Inpaint at 2X the image's size and 0.5 denoise
    async def superHirez(self, b64=None):

        # Only use the first image in discordFiles
        b64Image = b64.split(",",1)[0]
        image = getImageFormBase64(b64Image)
        image.save("output.png")
        # # Write the stuff
        # with open("output.png", "wb") as f:
        #     f.write(bytes.read())                        

        width, height = image.size
        mask = Image.new("RGB", (width, height), (1, 1, 1))
        maskB64 = getBase64FromImage(mask)

        payload = {
            "init_images": [ b64Image ], 
            "mask": maskB64, 
            "denoising_strength": 0.5, 
            "image_cfg_scale": 7, 
            "mask_blur": 10, 
            "inpaint_full_res": True,                   #choices=["Whole picture", "Only masked"]
            "inpaint_full_res_padding": 32, 
            "inpainting_mask_invert": 0,                #choices=['Inpaint masked', 'Inpaint not masked']
            "sampler_name": "DPM++ 2M Karras", 
            "batch_size": 1, 
            "steps": 30,
            "seed": self.seedToUse, 
            "cfg_scale": 7, 
            "width": width * 1.5, "height": height * 1.5, 
            "prompt": "masterpiece, best_quality, extremely detailed, intricate, high_details, sharp_focus , best_anatomy, hires, (colorful), beautiful, 4k, magical, adorable, (extraordinary:0.6), (((simple_background))), (((white_background))), multiple_views, reference_sheet, " + self.fixedPrompt, 
            "negative_prompt": "paintings, sketches, fingers, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, (outdoor:1.6), backlight,(ugly:1.331), (duplicate:1.331), (morbid:1.21), (mutilated:1.21), (tranny:1.331), mutated hands, (poorly drawn hands:1.5), blurry, (bad anatomy:1.21), (bad proportions:1.331), extra limbs, (disfigured:1.331), lowers, bad hands, missing fingers, extra digit, " + self.fixedNegative, 
            "sampler_index": "DPM++ 2M Karras"
        }
        
        file = open("output.txt", "w")
        json.dump(payload, file)
        file.close

        # # Add controlNet OpenPose
        # payload["alwayson_scripts"] = {
        #     "controlnet": {
        #         "args": [
        #             {
        #                 "enabled": True,
        #                 "input_image": b64Image,
        #                 "module": "openpose",
        #                 "model": "control_v11p_sd15_openpose [cab727d4]",
        #                 "weight": .75,  # Apply pose on 75% of the steps
        #                 "pixel_perfect": True
        #             },
        #             {
        #                 "enabled": True,
        #                 "input_image": b64Image,
        #                 "module": "depth_midas",
        #                 "model": "control_v11f1p_sd15_depth [cfd03158]",
        #                 "weight": 0.5, # Apply depth only 50% of the steps
        #                 "guidance": 1.0,
        #                 "guidance_start": 0.0,
        #                 "guidance_end": 0.5,
        #                 "pixel_perfect": True
        #             }
        #         ]
        #     }
        # }    

        # Prepare payload
        apiPath = "/sdapi/v1/img2img"

        # Sending API call request
        print(f"Sending request to {self.URL}{apiPath} ...")
        async with aiohttp.ClientSession(loop=self.ctx.bot.loop) as session:
            async with session.post(url=f'{self.URL}{apiPath}', json=payload)as response:
                print("Request returned: " + str(response.status))
                if response.status == 200:
                    r = await response.json()
                    # Extract the seed that was used to generate the image
                    info = r["info"]
                    infoJson = json.loads(info)
                    responseSeedUsed = infoJson["seed"]

                    # looping response to get actual image
                    discordFiles = []
                    for i in r['images']:

                        # Skip/Remove ControlNet images from output
                        if self.type == "inpaint" or self.type == "outfits" or self.type == "birth":
                            batchSize = payload["batch_size"]
                            if r["images"].index(i) >= batchSize: #Last 2 images are always ControlNet, Depth or OpenPose
                                continue # loop to next image

                        # Remove background
                        if self.removeBG:
                            i = await self.removeBackground(i, self.URL, self.ctx) if self.removeBG else i


                        # Image is in base64, convert it to a discord.File
                        bytes = io.BytesIO(base64.b64decode(i.split(",",1)[0]))
                        bytes.seek(0)
                        discordFile = discord.File(bytes, filename="{seed}-{ctx.message.author}.png")
                        discordFiles.append(discordFile)

                    # get available tags
                    tags = self.outputChanel.available_tags
                    # Prepare the tag to give to the new thread
                    forumTag = None
                    for tag in tags:
                        if tag.name.lower() == self.type.lower():
                            forumTag = tag
                            break

                    # Post on Discord Forum
                    await self.outputChanel.create_thread(
                        name=f"{self.type.upper()} by {self.ctx.message.author.display_name}", 
                        content=f"{self.ctx.author.mention} generated this image with prompt:\n```{self.ctx.message.content}``` and seed: {responseSeedUsed}", 
                        files=discordFiles,
                        applied_tags=[forumTag]
                    )
                
                else:
                    await self.ctx.send(f"{self.ctx.author.mention} -> API server returned an unknown error. Try again?")
                    print(response)