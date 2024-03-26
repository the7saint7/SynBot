import os
import io
import cv2
import json
import base64
import aiohttp
import asyncio
import discord
import requests
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from LORA_Helper import LORA_List
from discord.ext import tasks, commands
from charactersList import charactersLORA
from ssd_anime_face_detect import ssd_anime_face_detect_from_cv2_Image
from openPoses import getPose, getLewdPose, getImageAtPath, getBase64FromImage, getImageFormBase64, getBase64StringFromOpenCV

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

        # ControlNet
        self.enable_openPose = False
        self.enable_depth = False
        self.enable_softEdge = False
        
        # Common uploadedImages
        self.userBaseImage = None
        self.userControlNetImage = None
        
        # outfits
        self.outfitsCharacter = None
        self.outfitsPose = None
        self.outfitsName = None

        # birth
        self.birthPoses = []

        # expressions
        self.expressions = []
        self.includeBlush = False

        # # The message that was sent
        message = str(self.ctx.message.content)

        # Remove all possible prefixes
        prefixes = ["!Syn-txt2img", "!Syn-img2img", "!Syn-inpaint", "!Syn-outfits", "!Syn2-txt2img", "!Syn2-img2img", "!Syn2-inpaint", "!Syn2-outfits", "!Syn-birth", "!Syn2-birth", "!Syn-expressions", "!Syn2-expressions"]
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

        # ControlNet parameters
        # Some renders will require to read the passed image if controlnet is enabled
        if "controlNet" in jsonData:
            controlnet = jsonData["controlNet"]
            if "depth" in controlnet : self.enable_depth = True
            if "openPose" in controlnet : self.enable_openPose = True
            if "softEdge" in controlnet : self.enable_softEdge = True
        

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
            

            # Controlnet double-check
            if self.hasControlNet():
                attachmentCount = len(self.ctx.message.attachments)
                if attachmentCount == 0:
                    self.errorMsg = f"{self.ctx.author.mention} Missing a Image attachment for **controlNet** parameter"
                    self.isValid = False
                    return
                
                ###################### START lOADING USER SUBMITTED IMAGE
                self.loadUserSubmittedImages()
                self.userControlNetImage = self.userBaseImage
                if self.userControlNetImage == None:
                    self.errorMsg = "Could not read sent image. Request stopped."
                    self.isValid = False
                    return
                ###################### END LOADING USER SUBMITTED IMAGE


            
            self.format = "640x360" if self.formatStr.strip() == "landscape" else "360x640"
        ###### END TXT2IMG specific init

        ###### IMG2IMG specific init
        if self.type == "img2img":
            attachmentCount = len(self.ctx.message.attachments)

            # Do we have an attachment?
            if attachmentCount == 0:
                self.errorMsg = f"{self.ctx.author.mention} Missing a file attachment in IMG2IMG command"
                self.isValid = False
                return

            if "denoise" in jsonData:
                self.denoise = jsonData["denoise"]
            else:
                self.errorMsg = f"{self.ctx.author.mention} Missing **'denoise'** parameter"
                self.isValid = False
                return

            ###################### START lOADING USER SUBMITTED IMAGE
            self.loadUserSubmittedImages()
            if self.userBaseImage == None:
                self.errorMsg = "Could not read first image. Request stopped."
                self.isValid = False
                return
            ###################### END LOADING USER SUBMITTED IMAGE


        ###### END IMG2IMG specific init


        ###### INPAINT specific init
        if self.type == "inpaint":
            attachmentCount = len(self.ctx.message.attachments)
            if attachmentCount < 2:
                self.errorMsg = f"{self.ctx.author.mention} Missing **base Image** or **masked image** attachment in INPAINT command"
                self.isValid = False
                return

            if "denoise" in jsonData:
                self.denoise = jsonData["denoise"]
            else:
                self.errorMsg = f"{self.ctx.author.mention} Missing **'denoise'** parameter"
                self.isValid = False
                return

            ###################### START lOADING USER SUBMITTED BASE AND MASK IMAGE
            self.loadUserSubmittedImages()
            if self.userBaseImage == None or self.userControlNetImage == None:
                self.errorMsg = "Could not read sent images. Request stopped."
                self.isValid = False
                return
            ###################### END LOADING USER SUBMITTED BASE AND MASK IMAGE


        ###### END INPAINT specific init

        ###### OUTFITS specific init
        # Outfits do not support attachments. Use INPAINT for custom outfit-like execution
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
            self.userBaseImage = getImageAtPath(baseImagePath)
            
            maskImagePath = f"./sprites/{self.outfitsCharacter}/{self.outfitsPose}_{mask}.png"
            if not os.path.exists(maskImagePath):
                self.errorMsg = f"{self.ctx.author.mention} -> {maskImagePath} does not exist."
                self.isValid = False
                return
            self.userControlNetImage = getImageAtPath(maskImagePath)
            ###################### END DEFAULT BASE AND MASK IMAGE

        ###### END OUTFITS specific init

        ###### BIRTH specific init
        if self.type == "birth":
            
            # Default removeBG True if not specified in prompt parameters (will be overwritten a few lines bellow if specified in parameters)
            self.removeBG = True

            # 2 Modes: selecting poses (original), or passing a controlNet Image for the character pose(s)
            if self.hasControlNet():
                print ("ControlNet detected for BIRTH command")

                # using passed image as controlnet
                attachmentCount = len(self.ctx.message.attachments)

                # Do we have an attachment?
                if attachmentCount == 0:
                    self.errorMsg = f"{self.ctx.author.mention} Missing a **Image** attachment in BIRTH command"
                    self.isValid = False
                    return

                ###################### START lOADING USER SUBMITTED IMAGE
                self.loadUserSubmittedImages()
                # Twist, controlNet image is actually the baseImage
                self.userControlNetImage = self.userBaseImage

                if self.userControlNetImage == None:
                    self.errorMsg = "Could not read submitted image. Request stopped."
                    self.isValid = False
                    return
                ###################### END LOADING USER SUBMITTED IMAGE

            else:
                # using selected poses as controlnet
                print ("No controlNet for BIRTH command")

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
                
                self.userControlNetImage = getBase64FromImage(new_im)

                # Put our image in controlNet later
                self.enable_openPose = True


        ###### END BIRTH specific init
            
        ###### EXPRESSIONS specific init
        if self.type == "expressions":

            # Attachment logic:
                # if hasControlNet() -> Use the second image in controlnet
                # else use second image as mask
                # if no 2nd image, use as a normal IMG2IMG

            attachmentCount = len(self.ctx.message.attachments)
            if attachmentCount == 0:
                self.errorMsg = f"{self.ctx.author.mention} Missing **Image** attachment in EXPRESSIONS command"
                self.isValid = False
                return

            # Dont need a controlNet image, it's optional
            # if self.hasControlNet() and attachmentCount < 2:
            #     self.errorMsg = f"{self.ctx.author.mention} Missing **ControlNet Image** attachment in EXPRESSIONS command"
            #     self.isValid = False
            #     return

            # if not self.hasControlNet() and attachmentCount < 2:
            #     self.errorMsg = f"{self.ctx.author.mention} Missing **Mask Image** attachment in EXPRESSIONS command, OR include the **controlNet** parameter"
            #     self.isValid = False
            #     return
            
            if "expressions" in jsonData:
                self.expressions = jsonData["expressions"]
            
            if len(self.expressions) == 0:
                self.errorMsg = f"{self.ctx.author.mention} Missing **'expressions'** parameter"
                self.isValid = False
                return

            if "denoise" in jsonData:
                self.denoise = jsonData["denoise"]
            else:
                self.errorMsg = f"{self.ctx.author.mention} Missing **'denoise'** parameter"
                self.isValid = False
                return
            
            # Auto add Blush
            self.includeBlush = jsonData["includeBlush"] == "true" if "includeBlush" in jsonData else False

            # Re-using the inpaint image variables
            ###################### START lOADING USER SUBMITTED BASE AND MASK IMAGE
            self.loadUserSubmittedImages()
            if self.userBaseImage == None:
                self.errorMsg = "Could not read sent images. Request stopped."
                self.isValid = False
                return
            ###################### END LOADING USER SUBMITTED BASE AND MASK IMAGE
        ###### END EXPRESSIONS specific init

        # Optional parameters
        if "negative" in jsonData: self.userNegative = jsonData["negative"]
        if "hirez" in jsonData: self.hirez = jsonData["hirez"] == "true"
        if "batch" in jsonData: self.batchCount = jsonData["batch"]
        if "seed" in jsonData: self.seedToUse = jsonData["seed"]
        if "pose" in jsonData: self.poseNumber = jsonData["pose"]
        if "lewdPose" in jsonData: self.lewdPoseNumber = jsonData["lewdPose"]
        if "removeBG" in jsonData: self.removeBG = jsonData["removeBG"] == "true"
        if "lewdPose" in jsonData: self.lewdPoseNumber = jsonData["lewdPose"]
        # if "enableControlNet" in jsonData: self.enableControlNet= jsonData["enableControlNet"] == "true"

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


    # Utility function to load user images
    def loadUserSubmittedImages(self):

        attachmentCount = len(self.ctx.message.attachments)

        ###################### START lOADING USER SUBMITTED IMAGE
        print("Loading sent image as attachments...")

        # Base image is always first?
        baseImage =self.encode_discord_image(self.ctx.message.attachments[0].url, False)

        # Double check uploaded image dimension, resize if needed
        if baseImage.width > 1280 or baseImage.height > 1280:
            print("Uploaded image is too large, resizing...")
            baseImage.thumbnail((1280,1280), Image.Resampling.LANCZOS)
            print(f"Resized: {baseImage.width}, {baseImage.height}")
        
        # Back to base64
        self.userBaseImage = getBase64FromImage(baseImage)

        if attachmentCount > 1:
            ctrlNetImage =self.encode_discord_image(self.ctx.message.attachments[1].url, False) # controlNet?
            # Double check uploaded image dimension, resize if needed
            if ctrlNetImage.width > 1280 or ctrlNetImage.height > 1280:
                print("Uploaded image is too large, resizing...")
                ctrlNetImage.thumbnail((1280,1280), Image.Resampling.LANCZOS)
                print(f"Resized: {ctrlNetImage.width}, {ctrlNetImage.height}")
            
            # Back to base64
            self.userControlNetImage = getBase64FromImage(ctrlNetImage)
        imageCount = 1 if self.userBaseImage != None else 0
        imageCount += 1 if self.userControlNetImage != None else 0
        print(f"Attachments loaded in memory. ({imageCount})")
        ###################### END LOADING USER SUBMITTED IMAGE

    # Utility function to check all parameters at once
    def hasControlNet(self):
        return self.enable_depth or self.enable_openPose or self.enable_softEdge


    # Process an image URL and return a base64 encoded string
    def encode_discord_image(self, image_url, asBase64=True):
        try:
            response = requests.get(image_url)
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")

            if asBase64:
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
            else:
                return image
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
            
            # Where do we send the request?
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

            # Add/Enable openPose from selected pose or lewdPose image
            if poseImage != None:
                self.addControlNetToPayload(payload, poseImage, "openPose")
                # payload["alwayson_scripts"] = {
                #     "controlnet": {
                #         "args": [
                #             {
                #                 "input_image": poseImage,
                #                 "model": "control_v11p_sd15_openpose [cab727d4]",
                #                 "weight": 1,
                #                 # "width": 512,
                #                 # "height": 768,
                #                 "pixel_perfect": True
                #             }
                #         ]
                #     }
                # }
            
            # Add ControlNet if no pose/lewdPose and everything is in order
            elif self.hasControlNet() and self.userControlNetImage != None:
                if self.enable_depth:
                    self.addControlNetToPayload(payload, self.userControlNetImage, "depth")
                if self.enable_openPose:
                    self.addControlNetToPayload(payload, self.userControlNetImage, "openPose")
                if self.enable_softEdge:
                    self.addControlNetToPayload(payload, self.userControlNetImage, "softEdge")

        #########################         IMG2IMG        #####################################
        elif self.type == "img2img":

            # Where do we send the request?
            apiPath = "/sdapi/v1/img2img"

            pilImage = getImageFormBase64(self.userBaseImage)

            payload = {
                "init_images": [ self.userBaseImage ], 
                "denoising_strength": self.denoise, 
                "image_cfg_scale": 7, 
                "sampler_name": "DPM++ 2M Karras", 
                "batch_size": self.batchCount, 
                "steps": 30,
                "seed": self.seedToUse, 
                "cfg_scale": 7, 
                "width": pilImage.width, "height": pilImage.height, 
                "prompt": self.fixedPrompt, 
                "negative_prompt": self.fixedNegative, 
                "sampler_index": "DPM++ 2M Karras"
            }

            # Add ControlNet if requested in the parameters
            if self.hasControlNet():

                # Use controlNet image if one was passed, if not, baseImage
                imageToUse = self.userControlNetImage if self.userControlNetImage != None else self.userBaseImage

                if self.enable_depth:
                    self.addControlNetToPayload(payload, imageToUse, "depth")
                if self.enable_openPose:
                    self.addControlNetToPayload(payload, imageToUse, "openPose")
                if self.enable_softEdge:
                    self.addControlNetToPayload(payload, imageToUse, "softEdge")
            
        #########################        INPAINT       #####################################
        elif self.type == "inpaint":

            # Where do we send the request?
            apiPath = "/sdapi/v1/img2img"

            # This is how rescaled the inpainting will be. I should make this another parameter TODO
            width = 728
            height = 728
            if self.hirez:
                width = 856
                height = 856

            payload = {
                "init_images": [ self.userBaseImage ], 
                "mask": self.userControlNetImage,          # not really a controlnet, but the mask image
                "denoising_strength": self.denoise, 
                "image_cfg_scale": 7, 
                "mask_blur": 16, 
                "inpaint_full_res_padding": 32, 
                "inpaint_full_res": 1,                      # 0 - 'Whole picture' , 1 - 'Only masked' ||| True, # for 'Whole image' (The API doc has a mistake - the value must be a int - not a boolean)
                "inpainting_mask_invert": 0,                #choices=['Inpaint masked', 'Inpaint not masked']
                "initial_noise_multiplier": 1,              # I think this one is for inpaint models only. Leave it at 1 just in case. a simple noise multiplier but since we are setting the denoising_strength it seems unnecessary - recommended to leave it at 1. Once again the API doc is stupid with the 0 default that f*cks up results.
                "inpainting_fill": 1,                       # Value is int. 0 - 'fill', 1 - 'original', 2 - 'latent noise' and 3 - 'latent nothing'.
                "resize_mode": 1,                           # Crop and Resize
                "sampler_name": "DPM++ 2M Karras", 
                "batch_size": self.batchCount, 
                "steps": 30,
                "seed": self.seedToUse, 
                "cfg_scale": 7, 
                "width": width, "height": height, 
                "prompt": self.fixedPrompt, 
                "negative_prompt": self.fixedNegative, 
                "sampler_index": "DPM++ 2M Karras"
            }

            # Add ControlNet if requested in the parameters # DO NOT USE "userControlNetImage", it contains the MASK
            if self.hasControlNet():
                if self.enable_depth:
                    self.addControlNetToPayload(payload, self.userBaseImage, "depth")
                if self.enable_openPose:
                    self.addControlNetToPayload(payload, self.userBaseImage, "openPose")
                if self.enable_softEdge:
                    self.addControlNetToPayload(payload, self.userBaseImage, "softEdge")


        #########################        OUTFITS       #####################################
        elif self.type == "outfits":

            # Where do we send the request?
            apiPath = "/sdapi/v1/img2img"

            # This is how rescaled the inpainting will be.
            width = 728
            height = 728
            if self.hirez:
                width = 856
                height = 856

            # API Payload
            payload = {
                "init_images": [ self.userBaseImage ], 
                "mask": self.userControlNetImage, 
                "denoising_strength": self.denoise, 
                "image_cfg_scale": 7, 
                "mask_blur": 16, 
                "inpaint_full_res_padding": 32, 
                "inpaint_full_res": 1,                      # 0 - 'Whole picture' , 1 - 'Only masked' ||| True, # for 'Whole image' (The API doc has a mistake - the value must be a int - not a boolean)
                "inpainting_mask_invert": 0,                #choices=['Inpaint masked', 'Inpaint not masked']
                "initial_noise_multiplier": 1,              # I think this one is for inpaint models only. Leave it at 1 just in case. a simple noise multiplier but since we are setting the denoising_strength it seems unnecessary - recommended to leave it at 1. Once again the API doc is stupid with the 0 default that f*cks up results.
                "inpainting_fill": 1,                       # Value is int. 0 - 'fill', 1 - 'original', 2 - 'latent noise' and 3 - 'latent nothing'.
                "resize_mode": 1,                           # Crop and Resize
                "sampler_name": "DPM++ 2M Karras", 
                "sampler_name": "DPM++ 2M Karras", 
                "batch_size": self.batchCount, 
                "steps": 30,
                "seed": self.seedToUse, 
                "cfg_scale": 7, 
                "width": width, "height": height, 
                "prompt": self.fixedPrompt, 
                "negative_prompt": self.fixedNegative, 
                "sampler_index": "DPM++ 2M Karras"
            }

            # Add openPose as controlNet helper, should I also ad Depth? It might cause the outfit to "mold" to the body if using a nude image :shrug 
            self.addControlNetToPayload(payload, self.userBaseImage, "openPose")

            ##################################################################################
            # Outfits do not support attachments. Use INPAINT for custom outfit-like execution
            ##################################################################################
            # Add ControlNet if requested in the parameters
            # if self.hasControlNet():
            #     if self.enable_depth:
            #         self.addControlNetToPayload(payload, self.userBaseImage, "depth")
            #     if self.enable_openPose:
            #         self.addControlNetToPayload(payload, self.userBaseImage, "openPose")
            #     if self.enable_softEdge:
            #         self.addControlNetToPayload(payload, self.userBaseImage, "softEdge")

        #########################         BIRTH        #####################################
        elif self.type == "birth":
            
            # Where do we send the request?
            apiPath = "/sdapi/v1/txt2img"

            pilBaseImage = getImageFormBase64(self.userControlNetImage)
            payload = {
                "prompt": self.fixedPrompt,
                "negative_prompt": self.fixedNegative,
                "sampler_name": "DPM++ 2M Karras",
                "batch_size": self.batchCount,
                "steps": 35,
                "cfg_scale": 7,
                "width": pilBaseImage.width / 2,
                "height": pilBaseImage.height / 2,
                "restore_faces": False,
                "seed": self.seedToUse
            }

            # Hirez by 2 to recover from our small size
            payload["denoising_strength"] = 0.45
            payload["enable_hr"] = True
            payload["hr_upscaler"] = "4x-UltraSharp"
            payload["hr_scale"] = 2.0 if self.hirez == False else 3.0                         
            payload["hr_sampler_name"] = "DPM++ 2M Karras"
            payload["hr_second_pass_steps"] = 20
            
            # Add ControlNet if requested in the parameters
            if self.enable_depth:
                self.addControlNetToPayload(payload, self.userControlNetImage, "depth", preProcess=False)
            if self.enable_openPose:
                self.addControlNetToPayload(payload, self.userControlNetImage, "openPose", preProcess=False)
            if self.enable_softEdge:
                self.addControlNetToPayload(payload, self.userControlNetImage, "softEdge", preProcess=False)

        #########################       EXPRESSIONS      #####################################
        elif self.type == "expressions":
            
            # Attachment logic:
                # if hasControlNet() -> Use the second image in controlnet
                # else use second image as mask
                # if no 2nd image, or hasControlNet() --> use faceDetect to create mask
            
            # Base 64 to PIL Image
            pilImage = getImageFormBase64(self.userBaseImage)
            if pilImage == None:
                print("EMPTY PIL IMAGE in face-detect")

            masked_image = None
            cropped_mask_image_b64 = None # the final mask image

            # mask image is the second image being passed
            isSecondImageMask = True if not self.hasControlNet() and len(self.ctx.message.attachments) > 1 else False

            if isSecondImageMask:
                cropped_mask_image_b64 = self.userControlNetImage

            # we use face detect to create mask
            else:
                masked_image = Image.new('RGB', pilImage.size,  (1, 1, 1))

                ########################### EXTRACT FACES

                # PIL to OpenCV
                open_cv_image = np.array(pilImage, dtype=np.uint8)
                open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
                open_cv_image_mask = np.array(masked_image, dtype=np.uint8)
                open_cv_image_mask = cv2.cvtColor(open_cv_image_mask, cv2.COLOR_RGB2BGR)

                # From https://github.com/XavierJiezou/anime-face-detection?tab=readme-ov-file#repository-3
                model_path = "./model/ssd_anime_face_detect.pth"
                faces = ssd_anime_face_detect_from_cv2_Image(open_cv_image, model_path)

                # Loop faces and paint rectangle in white
                if len(faces) == 0:
                    self.errorMsg = f"{self.ctx.author.mention} No face detected in Image. Will have to process the whole image, results may take a long time and are probably going to be bad :man_shrugging:"
                    await self.ctx.send(self.errorMsg)

                else:
                    for k in range(faces.shape[0]):
                        xmin = int(faces[k, 0])
                        ymin = int(faces[k, 1])
                        xmax = int(faces[k, 2])
                        ymax = int(faces[k, 3])
                        ymin += 0.2 * (ymax - ymin + 1)
                        score = faces[k, 4]
                        # Fill the rectangle in white
                        cv2.rectangle(open_cv_image_mask, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 255), -1)
                    
                    # Convert back to base64
                    open_cv_image_mask = cv2.cvtColor(open_cv_image_mask, cv2.COLOR_BGR2RGB)
                    masked_image = Image.fromarray(open_cv_image_mask) 
                    masked_image.save("mask.png")
                    cropped_mask_image_b64 = getBase64FromImage(masked_image)

                ########################### END EXTRACT FACES


            ########################### END EXTRACT FACES TO MAKE RENDERING FASTER

            # Where do we send the request?
            apiPath = "/sdapi/v1/img2img"
            width, height = pilImage.size

            if cropped_mask_image_b64 != None:
                print ("EXPRESSION --> INPAINT")
                # INPAINT payload
                payload = {
                    "init_images": [ self.userBaseImage ], 
                    "mask": cropped_mask_image_b64, 
                    "denoising_strength": self.denoise, 
                    "image_cfg_scale": 7, 
                    "mask_blur": 16, 
                    "inpaint_full_res_padding": 32, 
                    "inpaint_full_res": 0,                      # 0 - 'Whole picture' , 1 - 'Only masked' ||| True, # for 'Whole image' (The API doc has a mistake - the value must be a int - not a boolean)
                    "inpainting_mask_invert": 0,                #choices=['Inpaint masked', 'Inpaint not masked']
                    "initial_noise_multiplier": 1,              # I think this one is for inpaint models only. Leave it at 1 just in case. a simple noise multiplier but since we are setting the denoising_strength it seems unnecessary - recommended to leave it at 1. Once again the API doc is stupid with the 0 default that f*cks up results.
                    "inpainting_fill": 1,                       # Value is int. 0 - 'fill', 1 - 'original', 2 - 'latent noise' and 3 - 'latent nothing'.
                    "resize_mode": 1,                           # Crop and Resize
                    "sampler_name": "DPM++ 2M Karras", 
                    "batch_size": 1, # no batch for you!
                    "steps": 35,
                    "seed": self.seedToUse, 
                    "cfg_scale": 7, 
                    "width": width, "height": height, 
                    "prompt": self.fixedPrompt, 
                    "negative_prompt": self.fixedNegative, 
                    "sampler_index": "DPM++ 2M Karras",
                    "script_name": "x/y/z plot",
                    "script_args": self.get_xyz_script_args(self.expressions),
                }

            else:
                print ("EXPRESSION --> IMG2IMG")
                # IMG2IMG payload || Should arrive here only if 2nd image is not mask and we couldn't detect a face
                payload = {
                    "init_images": [ self.userBaseImage ], 
                    "denoising_strength": self.denoise, 
                    "image_cfg_scale": 7, 
                    "sampler_name": "DPM++ 2M Karras", 
                    "batch_size": 1, # no batch for you!
                    "steps": 35,
                    "seed": self.seedToUse, 
                    "cfg_scale": 7, 
                    "width": width, "height": height, 
                    "prompt": self.fixedPrompt, 
                    "negative_prompt": self.fixedNegative, 
                    "sampler_index": "DPM++ 2M Karras",
                    "resize_mode": 1, # Crop and Resize
                    "script_name": "x/y/z plot",
                    "script_args": self.get_xyz_script_args(self.expressions),
                }

            # Either case, add openPose if needed
            print(f"{self.enable_depth}, {self.enable_openPose}, {self.enable_softEdge}")

            if self.enable_depth:
                self.addControlNetToPayload(payload, self.userBaseImage if isSecondImageMask else self.userControlNetImage, "depth")
            if self.enable_openPose:
                self.addControlNetToPayload(payload, self.userBaseImage if isSecondImageMask else self.userControlNetImage, "openPose")
            if self.enable_softEdge:
                self.addControlNetToPayload(payload, self.userBaseImage if isSecondImageMask else self.userControlNetImage, "softEdge")


        ######################### END PAYLOAD BUILDING #####################################
        
        self.printPayload(payload, toFile=True, shorten=False)

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
                    print(f"received {len(r['images'])} files")
                    for i in r['images']:

                        # expressions requests always return 2 grid images, skip them
                        if self.type == "expressions":
                            if r['images'].index(i) <= 1:
                                print("skipping returned file because expressions")
                                continue

                        # Skip/Remove ControlNet images from output
                        # if self.hasControlNet():
                        #     batchSize = payload["batch_size"]
                        #     if r["images"].index(i) >= batchSize: #Last 2 images are always ControlNet, Depth or OpenPose
                        #         continue # loop to next image

                        # Remove background
                        if self.removeBG:
                            if not (self.type == "birth" and self.hirez): # removing the BG before superHirez is a bad idea
                                # Do not removeBG on controlNet images
                                if not (r["images"].index(i) >= payload["batch_size"]):
                                    i = await self.removeBackground(i, self.URL, self.ctx) if self.removeBG else i

                        if len(discordFiles) < 10:
                            # Image is in base64, convert it to a discord.File
                            bytes = io.BytesIO(base64.b64decode(i.split(",",1)[0]))
                            bytes.seek(0)
                            discordFile = discord.File(bytes, filename="{seed}-{ctx.message.author}.png")
                            discordFiles.append(discordFile)


                    # print(f"showing {len(discordFiles)} files")
                    # # get available tags
                    # tags = self.outputChanel.available_tags
                    # # Prepare the tag to give to the new thread
                    # forumTag = None
                    # for tag in tags:
                    #     if tag.name.lower() == self.type.lower():
                    #         forumTag = tag
                    #         break

                    
                    # if the type is "birth" and "hirez", do a super-hirez on the image
                    if self.type == "birth" and self.hirez:
                        await self.superHirez(r['images'][0])
                    else:
                        await self.sendBotResponse(discordFiles, responseSeedUsed, payload)
                    
                else:
                    await self.ctx.send(f"{self.ctx.author.mention} -> API server returned an unknown error. Try again?")
                    print(response)
        
    async def getJOSNFromMessage(self):
        # Create the jsonData from the original prompt
        message = str(self.ctx.message.content)
        prefixes = ["!Syn-txt2img", "!Syn-img2img", "!Syn-inpaint", "!Syn-outfits", "!Syn2-txt2img", "!Syn2-img2img", "!Syn2-inpaint", "!Syn2-outfits", "!Syn-birth", "!Syn2-birth", "!Syn-expressions", "!Syn2-expressions"]
        for prefix in prefixes:
            message = message.removeprefix(prefix)

        # Try to load the JSON data, return if its invalid
        try:
            return json.loads(message.strip())
        except ValueError as e:
            self.errorMsg = f"{self.ctx.author.mention} invalid json: {e}"
            print(f"invalid json: {e} -> {message}")
            self.isValid = False
            return None
    
    async def sendBotResponse(self, discordFiles, responseSeedUsed, payload):

        # Create the jsonData from the original prompt
        jsonData = self.getJOSNFromMessage()


        
        # Create a list of messages, 1 for each image in discordFiles
        messages = []
        for dFile in discordFiles:
            seed = responseSeedUsed + discordFiles.index(dFile)
            jsonDataCopy = jsonData.copy()
            jsonDataCopy["seed"] = seed

            # Recreate prompt from jsonData
            command = (self.ctx.message.content).split(" ")[0] + " {\"hirez\":\"true\""
            command = command + ", \"seed\":" + seed + ", "
            if self.removeBG : command = command + ", \"removeBG\":\"true\""
            if self.hasControlNet() : command = command + ", \"controlNet\":[" + jsonDataCopy["controlNet"] + "]"
            if "denoise" in jsonData : command = command + ", \"denoise\":" + str(jsonDataCopy["denoise"])
            if "batch" in jsonData : command = command + ", \"batch\":" + str(jsonDataCopy["batch"])
            if "expressions" in jsonData : command = command + ", \"expressions\":[" + jsonDataCopy["expressions"] + "]"
            if "birthPoses" in jsonData : command = command + ", \"birthPoses\":[" + jsonDataCopy["birthPoses"] + "]"
            if "character" in jsonData : command = command + ", \"character\":\"" + jsonDataCopy["character"] + "\""
            if "outfit" in jsonData : command = command + ", \"outfit\":\"" + jsonDataCopy["outfit"] + "\""
            if "pose" in jsonData : command = command + ", \"pose\":\"" + jsonDataCopy["pose"] + "\""
            if "lewdPose" in jsonData : command = command + ", \"lewdPose\":\"" + jsonDataCopy["lewdPose"] + "\""
            if "format" in jsonData : command = command + ", \"format\":\"" + str(jsonDataCopy["format"]) + "\""
            if "prompt" in jsonData : command = command + ", \"prompt\":\"" + str(jsonDataCopy["prompt"]) + "\""
            if "negative" in jsonData : command = command + ", \"negative\":\"" + str(jsonDataCopy["negative"]) + "\""

            message = {
                "command": command,
                "file": dFile,
                "seed": seed
            }
            messages.append(message)



        # Add TAGS to thread
        # get available tags
        tags = self.outputChanel.available_tags
        # Prepare the tag to give to the new thread
        forumTag = None
        for tag in tags:
            if tag.name.lower() == self.type.lower():
                forumTag = tag
                break

        # Create the thread so we can use it to create the subsequent messages
        # Use the first message in the list for its content
        command1 = messages[0]["command"]
        thread = await self.outputChanel.create_thread(
            name=f"{self.type.upper()} by {self.ctx.message.author.display_name}", 
            content=f"{self.ctx.author.mention} generated this image with prompt:\n```{command1}``` and seed: {responseSeedUsed}", 
            files=discordFiles,
            applied_tags=[forumTag]
        )

        for message in messages:
            if messages.index(message) == 0:
                continue
            await thread.send(message["command"], discordFile=message["file"])


    def printPayload(self, payload, toFile=False, shorten=True) :
        payloadCopy = payload.copy()
        if shorten and "init_images" in payloadCopy: 
            for file in payloadCopy["init_images"]:
                payloadCopy["init_images"][payloadCopy["init_images"].index(file)] = file[:10] + "..."

            payloadCopy["init_images"] = str(payloadCopy["init_images"])[:10] + "..."
        if shorten and "mask" in payloadCopy : payloadCopy["mask"] = str(payloadCopy["mask"][:10]) + "..." 
        if shorten and "alwayson_scripts" in payloadCopy:
            for args in payloadCopy["alwayson_scripts"]["controlnet"]["args"]:
                args["input_image"] = str(args["input_image"][:10]) + "..."
        
        if toFile:
            file = open("output.txt", "w")
            json.dump(payloadCopy, file, indent=4)
            file.close
        else:
            print(payloadCopy)


    def addControlNetToPayload(self, payload, base64Image, module, preProcess=True):

        print(f"Preparing to add module: {module}...")
        if "alwayson_scripts" not in payload:
            payload["alwayson_scripts"] = {"controlnet": {
                "args": []
                }
            }

        script_payload = payload["alwayson_scripts"]["controlnet"]["args"]

        if module == "openPose":
            print("Adding openPose to payload")
            script_payload.append(
                {
                    "enabled": True,
                    "input_image": base64Image,
                    "module": "openpose",
                    "model": "control_v11p_sd15_openpose [cab727d4]",
                    "weight": .75,  # Apply pose on 75% of the steps
                    "pixel_perfect": True
                }
            )
        elif module == "depth":
            print("Adding depth to payload")
            script_payload.append(
                {
                    "enabled": True,
                    "input_image": base64Image,
                    "module": "depth_midas",
                    "model": "control_v11f1p_sd15_depth [cfd03158]",
                    "weight": 0.5, # Apply depth only 50% of the steps
                    "guidance": 1.0,
                    "guidance_start": 0.0,
                    "guidance_end": 0.5,
                    "pixel_perfect": True
                }
            )
        elif module == "softEdge":
            print("Adding softEdge to payload")
            script_payload.append(
                {
                    "enabled": True,
                    "input_image": base64Image,
                    "module": "softedge_pidinet",
                    "model": "control_v11p_sd15_softedge [a8575a2a]",
                    "weight": .5,  # Apply pose on 75% of the steps
                    "pixel_perfect": True
                }
            )

        # Remove pre-processor if not needed
        if not preProcess:
            print("Removing preprocessor...")
            script_payload[len(script_payload) -1]["module"] = None
        
        
        
        # = {
        #     "controlnet": {
        #         "args": [
        #             {
        #                 "enabled": True,
        #                 "input_image": base64Image,
        #                 "module": "openpose",
        #                 "model": "control_v11p_sd15_openpose [cab727d4]",
        #                 "weight": .75,  # Apply pose on 75% of the steps
        #                 "pixel_perfect": True
        #             },
        #             {
        #                 "enabled": True,
        #                 "input_image": base64Image,
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
            "mask_blur": 16, 
            "inpaint_full_res_padding": 32, 
            "inpaint_full_res": 1,                      # 0 - 'Whole picture' , 1 - 'Only masked' ||| True, # for 'Whole image' (The API doc has a mistake - the value must be a int - not a boolean)
            "inpainting_mask_invert": 0,                #choices=['Inpaint masked', 'Inpaint not masked']
            "initial_noise_multiplier": 1,              # I think this one is for inpaint models only. Leave it at 1 just in case. a simple noise multiplier but since we are setting the denoising_strength it seems unnecessary - recommended to leave it at 1. Once again the API doc is stupid with the 0 default that f*cks up results.
            "inpainting_fill": 1,                       # Value is int. 0 - 'fill', 1 - 'original', 2 - 'latent noise' and 3 - 'latent nothing'.
            "resize_mode": 1,                           # Crop and Resize
            "sampler_name": "DPM++ 2M Karras", 
            "sampler_name": "DPM++ 2M Karras", 
            "batch_size": 1, 
            "steps": 30,
            "seed": self.seedToUse, 
            "cfg_scale": 7, 
            "width": width * 1.5, "height": height * 1.5, 
            "prompt": self.fixedPrompt, 
            "negative_prompt": self.fixedNegative, 
            "sampler_index": "DPM++ 2M Karras"
        }
        
        # file = open("output.txt", "w")
        # json.dump(payload, file)
        # file.close

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
                        if self.hasControlNet():
                            batchSize = payload["batch_size"]
                            if r["images"].index(i) >= batchSize: #Last 2 images are always ControlNet, Depth or OpenPose
                                continue # loop to next image

                        # Before removing BG, add base image to output
                        bytes = io.BytesIO(base64.b64decode(i.split(",",1)[0]))
                        bytes.seek(0)
                        discordFile = discord.File(bytes, filename="{seed}-{ctx.message.author}.png")
                        discordFiles.append(discordFile)

                        # Remove background
                        if self.removeBG:
                            i = await self.removeBackground(i, self.URL, self.ctx) if self.removeBG else i

                        # Add removedBG image
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

    def get_xyz_script_args(self, expressions):

        print(expressions)

        # Customize x/y/z plot script parameters here
        # for AxisType index (8 for now) look at SD/scripts/xyz_grid.py
        # XAxis
        XAxisType = 7 # S/R Prompt
        XAxisValues = expressions
        XAxisDropdown = ""

        # YAxis
        YAxisType = 0 # Nothing
        YAxisValues = ""
        YAxisDropdown = ""

        # ZAxis
        ZAxisType = 0 # Nothing
        ZAxisValues = ""
        ZAxisDropdown = ""

        # The Rest
        drawLegend = "false"
        include_lone_images = "true"
        include_sub_grids = "false"
        no_fixed_seeds = "false"
        vary_seeds_x = "false"
        vary_seeds_y = "false"
        vary_seeds_z = "false"
        margin_size = 0
        csv_mode = "false"



        # def run(self, p, x_type, x_values, x_values_dropdown, y_type, y_values, y_values_dropdown, z_type, z_values, z_values_dropdown, draw_legend, include_lone_images, include_sub_grids, no_fixed_seeds, vary_seeds_x, vary_seeds_y, vary_seeds_z, margin_size, csv_mode):

        return [
            XAxisType, XAxisValues, XAxisDropdown,
            YAxisType, YAxisValues, YAxisDropdown, 
            ZAxisType, ZAxisValues, ZAxisDropdown, 
            drawLegend, 
            include_lone_images,
            include_sub_grids, 
            no_fixed_seeds, 
            vary_seeds_x,vary_seeds_y,vary_seeds_z,
            margin_size,
            csv_mode
        ]