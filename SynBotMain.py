import os
import io
import json
import base64
import aiohttp
import asyncio
import discord
import requests
from dotenv import load_dotenv
from LORA_Helper import LORA_List
from discord.ext import tasks, commands
from openPoses import getPose, getLewdPose
from charactersList import charactersLORA

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
    def __init__(self, context, outputChanel):
        self.ctx = context
        self.outputChanel = outputChanel

        # Default parameters
        self.isValid = True
        self.hirez = False
        self.hirezValue = 1.0
        self.seedToUse = -1
        self.batchCount = 1
        self.poseNumber = None
        self.lewdPoseNumber = None
        self.removeBG = False
        self.format = "640x360" 
        self.negative = "NEGATIVE"
        self.userPrompt = None
        self.URL = ""

        # # The message that was sent
        message = str(self.ctx.message.content)

        if message.startswith("!Syn-prompt") or message.startswith("!Syn2-prompt"):
            #Syn-prompt (NEW WAY)
            # message should be in format -> !Syn-prompt {"format": "landscape", "seed": 123456, "hirez": False, "batch": 4, "pose": 13, "lewdPose": 69, "removeBG": False, prompt": "my outfit prompt", "negative": "bad_anatomy"}
            jsonStr = message.removeprefix("!Syn-prompt").removeprefix("!Syn2-prompt").strip()
            jsonData = json.loads(jsonStr)

            # Required parameters
            if "prompt" in jsonData:
                self.userPrompt = jsonData["prompt"]
            else:
                self.isValid = False
                return

            if "format" in jsonData:
                self.formatStr = jsonData["format"]
            else:
                self.isValid = False
                return
            if self.formatStr != "landscape" and self.formatStr != "portrait":
                print("non conform format: " + self.formatStr)
                self.isValid = False
                return
            
            self.format = "640x360" if self.formatStr.strip() == "landscape" else "360x640"

            # Optional parameters
            if "negative" in jsonData: self.negative = jsonData["negative"]
            if "hirez" in jsonData: self.hirez = jsonData["hirez"] == "true"
            if "batch" in jsonData: self.batchCount = jsonData["batch"]
            if "seed" in jsonData: self.seedToUse = jsonData["seed"]
            if "pose" in jsonData: self.poseNumber = jsonData["pose"]
            if "lewdPose" in jsonData: self.lewdPoseNumber = jsonData["lewdPose"]
            if "removeBG" in jsonData: self.removeBG = jsonData["lewdPose"] == "true"

        else:
            #Syn-generate (OLD WAY)
            # Split the message with our delimiter - :
            # !Syn-generate : FORMAT : PROMPT : EXTRA-PARAMETERS
            data = message.split(":")
            if len(data) >= 3:
                # Ignore data[0]
                self.format = "640x360" if data[1].strip() == "landscape" else "360x640"
                self.userPrompt = data[2]
                parameters = data[3].split(",") if len(data) > 3 else []

                for paramData in parameters:
                    param = paramData.split("=")
                    if param[0].strip() == "hirez" and param[1].strip() == "true":
                        self.hirez = True
                    elif param[0].strip() == "seed":
                        self.seedToUse = param[1].strip()
                    elif param[0].strip() == "batch":
                        self.batchCount = int(param[1].strip())
                        if self.batchCount < 1 or self.batchCount > 4:
                            self.batchCount = 1
                    elif param[0].strip() == "removeBG" and param[1].strip() == "true":
                        self.removeBG = True
                    elif param[0].strip() == "pose":
                        self.poseNumber = param[1].strip()
                    elif param[0].strip() == "lewdPose":
                        self.lewdPoseNumber = param[1].strip()

            else:
                self.isValid = False

        # Reset batchCount if hirez
        if self.hirez and self.batchCount != 1:
            self.batchCount = 1

    async def generateImage(self):
        
        ######################### START PAYLOAD BUILDING #####################################
        # fix prompt by replacing character names with their LORAs
        fixedPrompt = self.userPrompt
        for key in charactersLORA.keys():
            if key in self.userPrompt:
                print("Found: " + key)
                fixedPrompt = fixedPrompt.replace(key, charactersLORA[key])

        # Same for LORA Helpers
        for key in LORA_List.keys():
            if key in self.userPrompt:
                print("Found: " + key + " in prompt")
                fixedPrompt = fixedPrompt.replace(key, LORA_List[key])
            elif key in self.negative:
                print("Found: " + key + " in negative")
                self.negative = self.negative.replace(key, LORA_List[key])

        payload = {
            "prompt": fixedPrompt,
            "negative_prompt": self.negative,
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
        ######################### END PAYLOAD BUILDING #####################################
        
        # Sending API call request
        print(f"Sending request to {self.URL} ...")
        async with aiohttp.ClientSession(loop=self.ctx.bot.loop) as session:
            async with session.post(url=f'{self.URL}/sdapi/v1/txt2img', json=payload)as response:
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
                        # Image is in base64, convert it to a discord.File
                        bytes = io.BytesIO(base64.b64decode(i.split(",",1)[0]))
                        bytes.seek(0)
                        discordFile = discord.File(bytes, filename="{seed}-{ctx.message.author}.png")
                        discordFiles.append(discordFile)

                    # Send a response with the image attached
                    await self.outputChanel.send(f"{self.ctx.author.mention} generated this image with prompt:{self.ctx.message.jump_url} and seed: {responseSeedUsed}", files=discordFiles)
        



