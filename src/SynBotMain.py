import os
import re
import io
import cv2
import math
import json
import base64
import random
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
from openPoses import getPose, getLewdPose, getImageAtPath, getBase64FromImage, getImageFormBase64, getBase64StringFromOpenCV, getSequencePose

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

        # Available format string
        self.availableFormatString = ["landscape", "portrait", "panno"]
        self.availableFormatSize = ["640x360", "360x640", "920x360"]

        self.availableSequenceTypes = ["Default", "Growth", "Shrink"]

        # Default parameters
        self.type = type # txt2img, img2img, inpaint, outfits, ect
        self.isValid = True
        self.hirez = False
        self.hirezValue = 1.0
        self.seedToUse = -1
        self.batchCount = 1
        self.poseNumber = None
        self.lewdPoseNumber = None
        self.removeBG = False
        self.formatIndex = 1 # See availableFormatString
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
        self.enable_reference = False
        
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

        # superHiRez
        self.customMaskBase64 = None
        self.scale = 2.0 # Default inpaint scale factor

        # sequence
        self.sequencePoses = [] # A lit of openPose images to use during the sequence
        self.startPrompt = ""
        self.endPrompt = ""
        self.commonPrompt = ""
        self.sequenceType = "Default"

        # Checkpoint
        # "cartoon" checkpoint: https://civitai.com/models/78306/cartoon-style
        self.checkpoint = None # Default AnyLora

        # The message that was sent
        message = str(self.ctx.message.content)

        # Remove all possible prefixes
        prefixes = ["!Syn-txt2img", "!Syn-img2img", "!Syn-inpaint", "!Syn-outfits", "!Syn2-txt2img", "!Syn2-img2img", "!Syn2-inpaint", "!Syn2-outfits", "!Syn-birth", "!Syn2-birth", "!Syn-expressions", "!Syn2-expressions", "!Syn-removeBG", "!Syn2-removeBG", "!Syn-superHiRez", "!Syn2-superHiRez", "!Syn-mask", "!Syn2-mask", "!Syn-sequence", "!Syn2-sequence"]
        for prefix in prefixes:
            message = message.removeprefix(prefix)

        # Try to load the JSON data, return if its invalid
        if len(message) == 0 or not message.find("{"): # removeBG does not contain any extra parameter
            print("No json data found in message, defaulting to empty jsonData")
            jsonData = []
        else:
            try:
                jsonData = json.loads(message.strip())
            except ValueError as e:
                fixedMessage = self.fixInput(message.strip())
                try:
                    jsonData = json.loads(fixedMessage)
                except ValueError as f:
                    self.errorMsg = f"{self.ctx.author.mention} invalid json: {f}"
                    print(f"invalid json: {f} -> {message}")
                    self.isValid = False
                    return

            # Required parameters
            if self.type == "removeBG" or self.type == "sequence":
                self.userPrompt = ""  #some types does not have a prompt
            elif "prompt" in jsonData:
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
            if "reference" in controlnet : self.enable_reference = True
        

        ###### removeBG specific init
        if self.type == "removeBG":
            attachmentCount = len(self.ctx.message.attachments)
            if attachmentCount == 0:
                self.errorMsg = f"{self.ctx.author.mention} Missing a Image attachment in **removeBG** command"
                self.isValid = False
                return
            
            ###################### START lOADING USER SUBMITTED IMAGE
            self.loadUserSubmittedImages()
            if self.userBaseImage == None:
                self.errorMsg = "Could not read sent image. Request stopped."
                self.isValid = False
                return
            ###################### END LOADING USER SUBMITTED IMAGE

        ###### mask specific init
        elif self.type == "mask":
            attachmentCount = len(self.ctx.message.attachments)
            if attachmentCount == 0:
                self.errorMsg = f"{self.ctx.author.mention} Missing a Image attachment in **superHiRez** command"
                self.isValid = False
                return
            
            ###################### START lOADING USER SUBMITTED IMAGE
            self.loadUserSubmittedImages()
            if self.userBaseImage == None:
                self.errorMsg = "Could not read sent image. Request stopped."
                self.isValid = False
                return
            ###################### END LOADING USER SUBMITTED IMAGE

        ###### superHiRez specific init
        elif self.type == "superHiRez":
            attachmentCount = len(self.ctx.message.attachments)
            if attachmentCount == 0:
                self.errorMsg = f"{self.ctx.author.mention} Missing a Image attachment in **superHiRez** command"
                self.isValid = False
                return
            
            ###################### START lOADING USER SUBMITTED IMAGE
            self.loadUserSubmittedImages()
            if self.userBaseImage == None:
                self.errorMsg = "Could not read sent image. Request stopped."
                self.isValid = False
                return
            ###################### END LOADING USER SUBMITTED IMAGE

            self.scale = jsonData["scale"] if "scale" in jsonData else 2.0 # Default 2.0
            if self.scale > 3.0 : self.scale = 3.0
            if self.scale < 1.2 : self.scale = 1.2


            # denoise is optional, no returning an error
            if "denoise" in jsonData:
                self.denoise = jsonData["denoise"]
            else:
                self.denoise = .50
                print(f"denoise parameter defaulting to 0.5")


        ###### SEQUENCE specific init
        elif self.type == "sequence":

            if "startPrompt" in jsonData:
                self.startPrompt = jsonData["startPrompt"]
            else:
                self.errorMsg = f"{self.ctx.author.mention} missing **startPrompt** parameter"
                print("missing **startPrompt** parameter")
                self.isValid = False
                return
            
            if "endPrompt" in jsonData:
                self.endPrompt = jsonData["endPrompt"]
            else:
                self.errorMsg = f"{self.ctx.author.mention} missing **endPrompt** parameter"
                print("missing **endPrompt** parameter")
                self.isValid = False
                return
            
            if "commonPrompt" in jsonData:
                self.commonPrompt = jsonData["commonPrompt"]
            
            if "sequencePoses" in jsonData:
                self.sequencePoses = jsonData["sequencePoses"]
                if len(self.sequencePoses) > 5:
                    self.errorMsg = f"{self.ctx.author.mention} too many poses in **sequencePoses**. Maximum of 5 poses allowed."
                    print("too many poses in **sequencePoses**")
                    self.isValid = False
                    return

            # else:
            #     self.errorMsg = f"{self.ctx.author.mention} missing **sequencePoses** parameter"
            #     print("missing **sequencePoses** parameter")
            #     self.isValid = False
            #     return

            if "sequenceType" in jsonData:
                self.sequenceType = jsonData["sequenceType"]
                if not self.sequenceType in self.availableSequenceTypes:
                    self.errorMsg = f"{self.ctx.author.mention} unknown **sequenceType**. Available sequenceTypes: {self.availableSequenceTypes}"
                    print("unknown **sequenceType**")
                    self.isValid = False
                    return
            
        ###### TXT2IMG specific init
        elif self.type == "txt2img":
            if "format" in jsonData:
                formatStr = jsonData["format"]

                if not formatStr in self.availableFormatString:
                    self.errorMsg = f"{self.ctx.author.mention} Non supported format: '{formatStr}'. Supported formats are: {self.availableFormatString}"
                    print("Non supported format: " + formatStr)
                    self.isValid = False
                    return

            else:
                self.errorMsg = f"{self.ctx.author.mention} Missing **'format'** parameter"
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


            self.formatIndex = self.availableFormatString.index(formatStr)

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

            self.scale = jsonData["scale"] if "scale" in jsonData else 1.0 # Default 1.0
            if self.scale > 3.0 : self.scale = 3.0
            if self.scale <= 0 : self.scale = 1.0

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
        if "checkpoint" in jsonData: 
            if jsonData["checkpoint"] == "cartoon":
                self.checkpoint = "722141adbc"
            elif jsonData["checkpoint"] == "hentai":
                self.checkpoint = "8145104977"
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
                
        # print(vars(self))


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
        return self.enable_depth or self.enable_openPose or self.enable_softEdge or self.enable_reference


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

    def has_transparency(self, img):
        if img.info.get("transparency", None) is not None:
            return True
        if img.mode == "P":
            transparent = img.info.get("transparency", -1)
            for _, index in img.getcolors():
                if index == transparent:
                    return True
        elif img.mode == "RGBA":
            extrema = img.getextrema()
            if extrema[3][0] < 255:
                return True

        return False
    
    def convertTransparentImageToMask(self, transparentImage):
        rgba = transparentImage.convert("RGBA")
        datas = rgba.getdata() 

        newData = [] 
        for item in datas: 
            if item[0] == 0 and item[1] == 0 and item[2] == 0:  # finding black colour by its RGB value 
                # storing a transparent value when we find a black colour 
                newData.append((1, 1, 1, 255)) 
            else: 
                newData.append((255, 255, 255, 255)) 

        rgba.putdata(newData) 
        return rgba

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
        #########################        removeBG           #####################################
        if self.type == "removeBG":
            
            # This one is special, we will call the async request right away and return when done

            baseImage = getImageFormBase64(self.userBaseImage)
            # Double check uploaded image dimension, resize if needed
            if baseImage.width > 1280 or baseImage.height > 1280:
                print("Uploaded image is too large, resizing...")
                baseImage.thumbnail((1280,1280), Image.Resampling.LANCZOS)
                print(f"Resized: {baseImage.width}, {baseImage.height}")
            
            baseImage64 = getBase64FromImage(baseImage)

            print("Removing background...")
            png = await self.removeBackground(baseImage64, self.URL, self.ctx)
            print("Background removed.")

            # Image is in base64, convert it to a discord.File
            bytes = io.BytesIO(base64.b64decode(png.split(",",1)[0]))
            bytes.seek(0)
            discordFile = discord.File(bytes, filename="removedBG-" + str(self.ctx.message.author) + ".png")

            # get available tags
            tags = self.outputChanel.available_tags
            # Prepare the tag to give to the new thread
            forumTag = None
            for tag in tags:
                if tag.name.lower() == self.type.lower():
                    forumTag = tag
                    break

            thread = await self.outputChanel.create_thread(
                name=f"Removed Background by {self.ctx.message.author.display_name}", 
                content=f"{self.ctx.author.mention} removed a background from the image here: {self.ctx.message.jump_url}", 
                file=discordFile,
                applied_tags=[forumTag]
            )

            return

        #########################           MASK            #####################################
        elif self.type == "mask":

            # base64 to PIL Image
            baseImage = getImageFormBase64(self.userBaseImage)
            # create a mask image of the same size but the mask should only cover the transparent pixels
            baseImageWithTransparency = baseImage if self.has_transparency(baseImage) else getImageFormBase64(await self.removeBackground(self.userBaseImage, self.URL, self.ctx))
            maskedImage = self.convertTransparentImageToMask(baseImageWithTransparency)

            # maskedImage = Image.new(mode="RGB", size=(baseImage.width, baseImage.height), color=(255,255,255))
            maskedImageB64 = getBase64FromImage(maskedImage)

            # No need to call any APY, send response right away
            discordFiles = []
            # Original
            bytes = io.BytesIO(base64.b64decode(self.userBaseImage))
            bytes.seek(0)
            discordFile = discord.File(bytes, filename="original.png")
            discordFiles.append(discordFile)
            # Mask
            bytes = io.BytesIO(base64.b64decode(maskedImageB64))
            bytes.seek(0)
            discordFile = discord.File(bytes, filename="mask.png")
            discordFiles.append(discordFile)

            # Dont create a forum thread, just reply in the same message
            await self.ctx.reply(f"Here is the masked image, {self.ctx.author.display_name}", files=discordFiles, mention_author=True)

            # We're done
            return


        #########################        SUPERHIREZ         #####################################
        elif self.type == "superHiRez":

            # Where do we send the request?
            apiPath = "/sdapi/v1/img2img"

            baseImage = getImageFormBase64(self.userBaseImage)

            # Double check uploaded image dimension, resize if needed
            if baseImage.width > 1280 or baseImage.height > 1280:
                print("Uploaded image is too large, resizing...")
                baseImage.thumbnail((1280,1280), Image.Resampling.LANCZOS)
                print(f"Resized: {baseImage.width}, {baseImage.height}")
            baseImage64 = getBase64FromImage(baseImage)

            # This is how rescaled the inpainting will be.
            width = baseImage.width * self.scale
            height = baseImage.height * self.scale

            # create a mask image of the same size but the mask should only cover the transparent pixels
            baseImageWithTransparency = baseImage if self.has_transparency(baseImage) else getImageFormBase64(await self.removeBackground(baseImage64, self.URL, self.ctx))
            maskedImage = self.convertTransparentImageToMask(baseImageWithTransparency)

            # maskedImage = Image.new(mode="RGB", size=(baseImage.width, baseImage.height), color=(255,255,255))
            maskedImageB64 = getBase64FromImage(maskedImage)
            self.customMaskBase64 = maskedImageB64 # Saved to we can include it in the response in Discord

            payload = {
                "init_images": [ baseImage64 ], 
                "mask": maskedImageB64,          # not really a controlnet, but the mask image
                "denoising_strength": self.denoise, 
                "image_cfg_scale": 7, 
                "mask_blur": 16, 
                "inpaint_full_res_padding": 32, 
                "inpaint_full_res": 1,                      # 0 - 'Whole picture' , 1 - 'Only masked' ||| True, # for 'Whole image' (The API doc has a mistake - the value must be a int - not a boolean)
                "inpainting_mask_invert": 0,                #choices=['Inpaint masked', 'Inpaint not masked']
                "initial_noise_multiplier": 1,              # I think this one is for inpaint models only. Leave it at 1 just in case. a simple noise multiplier but since we are setting the denoising_strength it seems unnecessary - recommended to leave it at 1. Once again the API doc is stupid with the 0 default that f*cks up results.
                "inpainting_fill": 1,                       # Value is int. 0 - 'fill', 1 - 'original', 2 - 'latent noise' and 3 - 'latent nothing'.
                "resize_mode": 1,                           # Crop and Resize
                "sampler_name": "DPM++ 2M", 
                "batch_size": 1, # No batch for you! 
                "steps": 50,
                "seed": self.seedToUse, 
                "cfg_scale": 7, 
                "width": width, "height": height, 
                "prompt": self.fixedPrompt, 
                "negative_prompt": self.fixedNegative, 
                "sampler_index": "DPM++ 2M"
            }

            # Add ControlNet if requested in the parameters
            if self.hasControlNet():
                if self.enable_depth:
                    self.addControlNetToPayload(payload, self.userBaseImage, "depth")
                if self.enable_openPose:
                    self.addControlNetToPayload(payload, self.userBaseImage, "openPose")
                if self.enable_softEdge:
                    self.addControlNetToPayload(payload, self.userBaseImage, "softEdge")
                if self.enable_reference:
                    self.addControlNetToPayload(payload, self.userBaseImage, "reference")


        #########################        TXT2IMG         #####################################
        elif self.type == "txt2img":
            
            format = self.availableFormatSize[self.formatIndex]

            # Where do we send the request?
            apiPath = "/sdapi/v1/txt2img"

            payload = {
                "prompt": self.fixedPrompt,
                "negative_prompt": self.fixedNegative,
                "sampler_name": "DPM++ 2M",
                "batch_size": self.batchCount,
                "steps": 35,
                "cfg_scale": 7,
                "width": int(format.split("x")[0]),
                "height": int(format.split("x")[1]),
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
                payload["hr_sampler_name"] = "DPM++ 2M"
                payload["hr_second_pass_steps"] = 20
            
            poseImage = None
            if self.poseNumber != None:
                # Pick a pose according toe format and "shot"
                pose_format = self.availableFormatString[self.formatIndex]
                pose_shot = "full_body" if "full_body" in self.userPrompt else "cowboy_shot"
                poseImage = getPose(pose_format, pose_shot, self.poseNumber)

            if self.lewdPoseNumber != None:
                # Pick a pose according toe format and "shot"
                poseImage = getLewdPose(self.lewdPoseNumber)

            # Add/Enable openPose from selected pose or lewdPose image
            if poseImage != None:
                self.addControlNetToPayload(payload, poseImage, "openPose", preProcess=False)
            
            # Add ControlNet if no pose/lewdPose and everything is in order
            elif self.hasControlNet() and self.userControlNetImage != None:
                if self.enable_depth:
                    self.addControlNetToPayload(payload, self.userControlNetImage, "depth", preProcess=False)
                if self.enable_openPose:
                    self.addControlNetToPayload(payload, self.userControlNetImage, "openPose", preProcess=False)
                if self.enable_softEdge:
                    self.addControlNetToPayload(payload, self.userControlNetImage, "softEdge", preProcess=False)
                if self.enable_reference:
                    self.addControlNetToPayload(payload, self.userControlNetImage, "reference")

        #########################         IMG2IMG        #####################################
        elif self.type == "img2img":

            # Where do we send the request?
            apiPath = "/sdapi/v1/img2img"

            pilImage = getImageFormBase64(self.userBaseImage)

            payload = {
                "init_images": [ self.userBaseImage ], 
                "denoising_strength": self.denoise, 
                "image_cfg_scale": 7, 
                "sampler_name": "DPM++ 2M", 
                "batch_size": self.batchCount, 
                "steps": 30,
                "seed": self.seedToUse, 
                "cfg_scale": 7, 
                "width": pilImage.width, "height": pilImage.height, 
                "prompt": self.fixedPrompt, 
                "negative_prompt": self.fixedNegative, 
                "sampler_index": "DPM++ 2M"
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
                if self.enable_reference:
                    self.addControlNetToPayload(payload, imageToUse, "reference")
            
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
            
            # If scale is present, use the baseImage real's size but scaled up or down
            if self.scale != 1.0:
                image = getImageFormBase64(self.userBaseImage)
                width = image.width * self.scale
                height = image.height * self.scale

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
                "sampler_name": "DPM++ 2M", 
                "batch_size": self.batchCount, 
                "steps": 30,
                "seed": self.seedToUse, 
                "cfg_scale": 7, 
                "width": width, "height": height, 
                "prompt": self.fixedPrompt, 
                "negative_prompt": self.fixedNegative, 
                "sampler_index": "DPM++ 2M"
            }

            # Add ControlNet if requested in the parameters # DO NOT USE "userControlNetImage", it contains the MASK
            if self.hasControlNet():
                if self.enable_depth:
                    self.addControlNetToPayload(payload, self.userBaseImage, "depth")
                if self.enable_openPose:
                    self.addControlNetToPayload(payload, self.userBaseImage, "openPose")
                if self.enable_softEdge:
                    self.addControlNetToPayload(payload, self.userBaseImage, "softEdge")
                if self.enable_reference:
                    self.addControlNetToPayload(payload, self.userBaseImage, "reference")


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
                "sampler_name": "DPM++ 2M", 
                "sampler_name": "DPM++ 2M", 
                "batch_size": self.batchCount, 
                "steps": 30,
                "seed": self.seedToUse, 
                "cfg_scale": 7, 
                "width": width, "height": height, 
                "prompt": self.fixedPrompt, 
                "negative_prompt": self.fixedNegative, 
                "sampler_index": "DPM++ 2M"
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
                "sampler_name": "DPM++ 2M",
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
            payload["hr_sampler_name"] = "DPM++ 2M"
            payload["hr_second_pass_steps"] = 20
            
            # Add ControlNet if requested in the parameters
            if self.enable_depth:
                self.addControlNetToPayload(payload, self.userControlNetImage, "depth", preProcess=False)
            if self.enable_openPose:
                self.addControlNetToPayload(payload, self.userControlNetImage, "openPose", preProcess=False)
            if self.enable_softEdge:
                self.addControlNetToPayload(payload, self.userControlNetImage, "softEdge", preProcess=False)
            if self.enable_reference:
                self.addControlNetToPayload(payload, self.userControlNetImage, "reference")

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
                    # masked_image.save("mask.png")
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
                    "sampler_name": "DPM++ 2M", 
                    "batch_size": 1, # no batch for you!
                    "steps": 35,
                    "seed": self.seedToUse, 
                    "cfg_scale": 7, 
                    "width": width, "height": height, 
                    "prompt": self.fixedPrompt, 
                    "negative_prompt": self.fixedNegative, 
                    "sampler_index": "DPM++ 2M",
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
                    "sampler_name": "DPM++ 2M", 
                    "batch_size": 1, # no batch for you!
                    "steps": 35,
                    "seed": self.seedToUse, 
                    "cfg_scale": 7, 
                    "width": width, "height": height, 
                    "prompt": self.fixedPrompt, 
                    "negative_prompt": self.fixedNegative, 
                    "sampler_index": "DPM++ 2M",
                    "resize_mode": 1, # Crop and Resize
                    "script_name": "x/y/z plot",
                    "script_args": self.get_xyz_script_args(self.expressions),
                }

            if self.hasControlNet():
                if self.enable_depth:
                    self.addControlNetToPayload(payload, self.userBaseImage if isSecondImageMask else self.userControlNetImage, "depth")
                if self.enable_openPose:
                    self.addControlNetToPayload(payload, self.userBaseImage if isSecondImageMask else self.userControlNetImage, "openPose")
                if self.enable_softEdge:
                    self.addControlNetToPayload(payload, self.userBaseImage if isSecondImageMask else self.userControlNetImage, "softEdge")
                if self.enable_reference:
                    self.addControlNetToPayload(payload, self.userBaseImage if isSecondImageMask else self.userControlNetImage, "reference")

        #########################       SEQUENCES        #####################################
        elif self.type == "sequence":
            
            # Build a list of all the images in the sequence
            poseImages = []
            for pose in self.sequencePoses:
                poseImages.append(getSequencePose(pose, asPIL=True))

            # # Grow or shrink images
            # if self.sequenceType == "Growth":
            #     print("TODO: Growth")
            # elif self.sequenceType == "Shrink":
            #     print("TODO: Shrink")

            ################# Fix the starting and ending prompt
            fixedStartPrompt = self.startPrompt
            fixedEndPrompt = self.endPrompt
            fixedCommon = self.commonPrompt

            for key in charactersLORA.keys():
                found = False
                if key in fixedStartPrompt:
                    print("Found: " + key)
                    found = True
                    fixedStartPrompt = fixedStartPrompt.replace(key, charactersLORA[key])
                if key in fixedEndPrompt:
                    if not found:
                        print("Found: " + key)
                    fixedEndPrompt = fixedEndPrompt.replace(key, charactersLORA[key])
                if key in fixedCommon:
                    if not found:
                        print("Found: " + key)
                    fixedCommon = fixedCommon.replace(key, charactersLORA[key])

            # Same for LORA Helpers
            for key in LORA_List.keys():
                if key in fixedStartPrompt:
                    print("Found: " + key + " in prompt")
                    fixedStartPrompt = fixedStartPrompt.replace(key, LORA_List[key])
                if key in fixedEndPrompt:
                    print("Found: " + key + " in negative")
                    fixedEndPrompt = fixedEndPrompt.replace(key, LORA_List[key])
                if key in fixedCommon:
                    print("Found: " + key + " in negative")
                    fixedCommon = fixedCommon.replace(key, LORA_List[key])
            ################# END Fix the starting and ending prompt

            # Removing logic to get common prompt as the user will now provide the common prompt
            # fixedStartPrompt2 = fixedStartPrompt
            # fixedEndPrompt2 = fixedEndPrompt
            # startTags = [x.strip() for x in fixedStartPrompt.split(',')]
            # endTags = [x.strip() for x in fixedEndPrompt.split(',')]
            # commonTags = []
            # for tag in startTags:
            #     if tag in endTags:
            #         fixedStartPrompt2 = fixedStartPrompt2.replace(tag, "")
            #         fixedEndPrompt2 = fixedEndPrompt2.replace(tag, "")
            #         if tag.strip() != "":
            #             commonTags.append(tag.strip())
            
            # print(f"common tags: {commonTags}")

            # We are going to make 5 poses so 5 prompts, but we might have 1 to 5 images. We must split the poses evenly
            prompts = []
            for promptIndex in range(5):
                prompt = {
                    "prompt": self.getPromptForSequence(fixedStartPrompt, fixedEndPrompt, fixedCommon, promptIndex),
                    "image": None if len(poseImages) == 0 else poseImages[math.floor(len(poseImages) / 5 * promptIndex)]
                }
                prompts.append(prompt)
            
            # We need to loop 5 times and call SD in each loop with all the prompt info
            await self.createSequence(prompts)
            return


        ############################# CHECKPOINT #########################################
        # Override settings for custom checkpoint
        if self.checkpoint != None:
            print(f"Switching checkpoint to: {self.checkpoint}")
            payload["override_settings"] = {
                "sd_model_checkpoint": self.checkpoint
            }

        ######################### END PAYLOAD BUILDING #####################################
        
        # print(payload)
        # self.printPayload(payload, toFile=True, shorten=False)

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
                            discordFile = discord.File(bytes, filename="" + str(responseSeedUsed) + "-" + str(self.ctx.message.author) + ".png")
                            discordFiles.append(discordFile)

                    # superHiRez, include the original image and the custom mask
                    if self.type == "superHiRez":
                        # Original
                        bytes = io.BytesIO(base64.b64decode(self.userBaseImage))
                        bytes.seek(0)
                        discordFile = discord.File(bytes, filename="original.png")
                        discordFiles.append(discordFile)
                        # Mask
                        bytes = io.BytesIO(base64.b64decode(self.customMaskBase64))
                        bytes.seek(0)
                        discordFile = discord.File(bytes, filename="mask.png")
                        discordFiles.append(discordFile)
                        

                    print(f"showing {len(discordFiles)} files")
                    # get available tags
                    tags = self.outputChanel.available_tags
                    # Prepare the tag to give to the new thread
                    forumTag = None
                    for tag in tags:
                        if tag.name.lower() == self.type.lower():
                            forumTag = tag
                            break

                    
                    # if the type is "birth" and "hirez", do a super-hirez on the image
                    if self.type == "birth" and self.hirez:
                        await self.superHirez(r['images'][0])
                        return



                    thread = await self.outputChanel.create_thread(
                        name=f"{self.getTitle()} by {self.ctx.message.author.display_name}", 
                        content=f"{self.ctx.author.mention} generated this image with prompt: {self.ctx.message.jump_url}\n```{self.getPromptWithSeed(responseSeedUsed)}```", 
                        files=discordFiles,
                        applied_tags=[forumTag]
                    )

                    # # Add reaction for Bot to detect
                    # await thread.message.add_reaction('🔃')
                
                else:
                    await self.ctx.send(f"{self.ctx.author.mention} -> API server returned an unknown error. Try again?")
                    print(response)
                    print(json.dumps(payload))
        

    def getPromptWithSeed(self, seed):
        prompt = str(self.ctx.message.content)
        if not "seed" in prompt:
            print("Seed added to prompt")
            prompt = prompt.replace(" {", " {\"seed\":" + str(seed) + ", ")
        else:
            print("Seed already in prompt")
        
        return prompt

    def getTitle(self):

        exceptions = ["QUALITY", "NEGATIVE"]
        title = ""


        for key in charactersLORA.keys():
            if key in exceptions:
                continue 
            if key in self.userPrompt:
                title = title + " " + key
        for key in LORA_List.keys():
            if key in exceptions:
                continue 
            if key in self.userPrompt:
                title = title + " " + key

        # Default, type uppercase
        if len(title) == 0:
            title = self.type.upper()

        print("Title: " + self.userPrompt)
        return title.strip()

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
            json.dumps(payloadCopy, file, indent=4)
            file.close
        else:
            print(payloadCopy)


    def addControlNetToPayload(self, payload, base64Image, module, preProcess=True):

        # print(f"Preparing to add module: {module}...")
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
        elif module == "reference":
            print("Adding reference to payload")
            script_payload.append(
                {
                    "enabled": True,
                    "input_image": base64Image,
                    "module": "reference_only",
                    "model": "none",
                    # "weight": 1.5,
                    # "guidance": 1.0,
                    # "guidance_start": 0.0,
                    # "guidance_end": 1.0,
                    # "resize_mode": 1   # Crop and Resize
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
        # image.save("output.png")
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
            "sampler_name": "DPM++ 2M", 
            "sampler_name": "DPM++ 2M", 
            "batch_size": 1, 
            "steps": 30,
            "seed": self.seedToUse, 
            "cfg_scale": 7, 
            "width": width * 1.5, "height": height * 1.5, 
            "prompt": self.fixedPrompt, 
            "negative_prompt": self.fixedNegative, 
            "sampler_index": "DPM++ 2M"
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
                        name=f"{self.getTitle()} by {self.ctx.message.author.display_name}", 
                        content=f"{self.ctx.author.mention} generated this image with prompt:\n```{self.getPromptWithSeed(responseSeedUsed)}```", 
                        files=discordFiles,
                        applied_tags=[forumTag]
                    )
                
                else:
                    await self.ctx.send(f"{self.ctx.author.mention} -> API server returned an unknown error. Try again?")
                    print(response)
                    print(json.dumps(payload))

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
    
    # input bad jason string, tries to fix by replacing keys with string keys
    def fixInput(self, message):
        
        keys = ["format:", "batch:", "hirez:", "prompt:", "negative:", "controlNet:", "pose:", "lewdPose:", "birthPose:", "scale:", "seed:", "removeBG:", "denoise:", "character:", "outfit:", "expressions:", "sequence:", "startPrompt:", "endPrompt:", "sequencePoses:", "sequenceType:", "commonPrompt:", "checkpoint:"]
        fixed = message
        for key in keys:
            if key in fixed:
                fixed = fixed.replace(key, "\""+ key.replace(":", "") + "\":")


        return fixed

    # For sequences, adjust weights as the prompt progress
    def getPromptForSequence(self, start, end, common, index):

        startWeights = [1.0, .75, 0.5, 0.25, 0.0]
        endWeights = [0.0, 0.25, 0.5, 0.75, 1.0]

        # split each prompt into words and add a weight to each word
        startWords = start.split(",")
        startFixed = ""
        startWeight = startWeights[index]
        for word in startWords:
            # more complicated with LORA
            if word.strip().startswith("<lora:"):
                lora = word.strip().split(":")
                maxWeight = float(lora[2].replace(">", ""))
                correctedWeight = maxWeight * startWeights[index]
                startFixed += f"<lora:{lora[1]}:{correctedWeight}>,"
            else:
                if word.strip() != "":
                    startFixed += f"({word.strip()}:{startWeight}),"

        endWords = end.split(",")
        endFixed = ""
        endWeight = endWeights[index]
        for word in endWords:
            # Ignore LORA
            if word.strip().startswith("<lora:"):
                lora = word.strip().split(":")
                maxWeight = float(lora[2].replace(">", ""))
                correctedWeight = maxWeight * endWeights[index]
                endFixed += f"<lora:{lora[1]}:{correctedWeight}>,"
            else:
                if word.strip() != "":
                    endFixed += f"({word.strip()}:{endWeight}),"

        return common + ", " + startFixed + endFixed.rstrip(',')
    
    async def createSequence(self, prompts):
        
        responseImages = []

        # Use the same seed for each images
        seed = self.seedToUse if self.seedToUse !=-1 else random.randint(0,4294967295)
        print(f"seed: {seed}")

        for prompt in prompts:

            print(prompt["prompt"])

            # Build payload
            payload = {
                "prompt": prompt["prompt"],
                "negative_prompt": self.fixedNegative,
                "sampler_name": "DPM++ 2M",
                "batch_size": self.batchCount,
                "steps": 35,
                "cfg_scale": 7,
                "width": 512,
                "height": 768,
                "restore_faces": False,
                "seed": seed
            }

            if self.hirez:
                payload["denoising_strength"] = 0.5
                payload["enable_hr"] = True
                payload["hr_upscaler"] = "4x-UltraSharp"
                #payload["hr_resize_x"] = int(format.split("x")[0]) * HIREZ_SCALE
                #payload["hr_resize_y"] = int(format.split("x")[1]) * HIREZ_SCALE
                payload["hr_scale"] = self.hirezValue
                payload["hr_sampler_name"] = "DPM++ 2M"
                payload["hr_second_pass_steps"] = 20

            if prompt["image"] != None:
                self.addControlNetToPayload(payload, getBase64FromImage(prompt["image"]), "openPose", preProcess=False)

            # end Build payload

            # Prepare payload
            apiPath = "/sdapi/v1/txt2img"

            # Sending API call request
            print(f"Sending request to {self.URL}{apiPath} ...")

            # response = requests.post(url=f'{self.URL}{apiPath}', json=payload)
            # print(response)
            # print("Request returned: " + str(response.status_code))
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
                        responseImages.append(r['images'][0])

                        print("Sequence image: " + str(len(responseImages)))
            
        print("Sequence completed")

        # Turn all those images into discordFile and send them
        discordFiles = []
        for base64Image in responseImages:
            # Image is in base64, convert it to a discord.File
            bytes = io.BytesIO(base64.b64decode(base64Image.split(",",1)[0]))
            bytes.seek(0)
            discordFile = discord.File(bytes, filename="{seed}-{ctx.message.author}.png")
            discordFiles.append(discordFile)

        # stich all the images into a single wide image and insert it as the first image
        pilImages = []
        for base64Imag in responseImages:
            pilImages.append(getImageFormBase64(base64Imag))
        # Concatenate the images
        widths, heights = zip(*(i.size for i in pilImages))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in pilImages:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0]
        
        # Convert stitched image to discordFile
        base64StitchedImage = getBase64FromImage(new_im)
        bytes = io.BytesIO(base64.b64decode(base64StitchedImage))
        bytes.seek(0)
        discordFile = discord.File(bytes, filename="stitched.png")
        discordFiles.insert(0, discordFile)

        # get available tags
        tags = self.outputChanel.available_tags

        # Prepare the tag to give to the new thread
        forumTag = None
        for tag in tags:
            if tag.name.lower() == self.type.lower():
                forumTag = tag
                break

        # Post on Discord Forum
        # Only send the stitched image
        thread = await self.outputChanel.create_thread(
            name=f"{self.getTitle()} by {self.ctx.message.author.display_name}", 
            content=f"{self.ctx.author.mention} generated this image with prompt:\n```{self.getPromptWithSeed(responseSeedUsed)}```", 
            file=discordFiles[0],
            applied_tags=[forumTag]
        )

        for prompt in prompts:
            promptText = prompt["prompt"]
            await thread.thread.send(f"```{promptText}```", file=discordFiles[prompts.index(prompt) +1])


        