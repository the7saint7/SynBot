import os
import io
import sys
import json
import base64
import aiohttp
import asyncio
import discord
import requests
from PIL import Image
from dotenv import load_dotenv
from discord.ext import commands
from LORA_Helper import LORA_List
from charactersList import charactersLORA
from SynBotMain import SynBotManager, SynBotPrompt
from openPoses import getPose, getLewdPose, getImageAtPath

# syn2: The plan is to run 2 bots, because I have 2 StableDiffusion server, this would help with the workload
# Users will be able to select which bot to work with
# env_dev: This will fix the channels to use to be private for ToS Admin role only

load_dotenv()
env_dev = False
syn2 = False
if len(sys.argv) > 1:
    env_dev = True if "dev" in sys.argv else False
    if env_dev:
        print("Starting server in dev mode")
    syn2 = True if "syn2" in sys.argv else False

# Printing the Bot name
if syn2:
    print("Starting Syn-Bot 2")
else:
    print("Starting Syn-Bot 1")

def getAPIURL():
    if syn2:
        return os.getenv("SD_API_URL2")
    else:
        return os.getenv("SD_API_URL1")

# Discord Bot
if syn2:
    bot = SynBotManager(command_prefix="!Syn2-", intents=discord.Intents.all())
else:
    bot = SynBotManager(command_prefix="!Syn-", intents=discord.Intents.all())

@bot.event
async def on_ready():
    aiNumInstruction = "2" if syn2 else ""
    channel_id = int(os.getenv("DEV_CHANNEL")) if env_dev else int(os.getenv("INPUT_CHANEL"))
    channel = bot.get_channel(channel_id)
    await channel.send(f"**Syn-Bot{aiNumInstruction} AI** is online and ready to generate some images. Type **!Syn{aiNumInstruction}-helpMe** for instructions!")

@bot.command()
async def helpMe(ctx):
    aiNumInstruction = "2" if syn2 else ""
    await ctx.message.author.send(f"You can ask me to generate ST-related images using these keywords: !Syn{aiNumInstruction}-generate : FORMAT[landscape|portrait]: prompt with names like JOHN, KATRINA, in UPPERCASE\n**!Syn{aiNumInstruction}-generate : landscape: JOHN as a girl at a beach, wearing a bikini**\n**!Syn{aiNumInstruction}-generate : portrait: KATRINA wearing a school uniform in a classroom yelling at someone else**\nYou can get a full list of supported ST character names by using the command **!Syn{aiNumInstruction}-characters**\nYou can get a list of advanced parameters by calling **!Syn{aiNumInstruction}-advanced**\nYou can get a list pre-formated LORAs by calling **!Syn{aiNumInstruction}-loras**\nYou can also check the list of available poses in the **syn-ai-help** forum")

@bot.command()
async def advanced(ctx):
    aiNumInstruction = "2" if syn2 else ""
    await ctx.message.author.send(f"Here is a list of extra parameters you can add at the end of your prompt. Make sure you add an extra ':' to separate your prompt from the extra parameters.\n**hirez=true**: Will increase the resolution of the image and make it look sharper. This feature could be disabled if abused as it taxes my PC a lot.\n**removeBG=true**: Will try to remove the background from the image. Results will vary.\n**seed=654789**: You can enter any seed NUMBER yo want. This way you can change your prompt a little and use the same seed, to see how it changes.\n**batch=4**: will create 4 images, you can use any number between 1 and 4. batch count will be ignore if hirez is True.\nA neat trick, is to NOT use hirez, until you find an image that you like, then you can use the same prompt, pass it the hirez and seed number and it will generate the same image but in higher quality & resolution.\n\nExample:\n**!Syn{aiNumInstruction}-generate : portrait: JOHN, TKUNIFORM, a girl wearing a school uniform in a classroom angry, frown, hands_on_own_hips, hand_on_hip, large_breasts, long_hair, ponytail : hirez=true, seed=3955923732**")

@bot.command()
async def characters(ctx):
    characterNames = "\n".join(charactersLORA.keys())
    await ctx.message.author.send(f"Here's a list of ST character names you can use in your prompt. Remember to write them in uppercase.\n{characterNames}")

@bot.command()
async def loras(ctx):
    list = "\n".join(LORA_List.keys())
    await ctx.message.author.send(f"Here's a list of extra tags that will be converted to some lora preset I've compiled for you.\n{list}")

@bot.command()
async def generate(ctx):

    # Select proper channel to handle requests
    input_channel_id = int(os.getenv("DEV_CHANNEL")) if env_dev else int(os.getenv("INPUT_CHANEL"))

    # Ignore generate request if it's not coming from the right channel
    inputChannel = bot.get_channel(input_channel_id)
    if ctx.channel != inputChannel:
        print ("Request made in wrong channel")
        return

    output_channel_id = int(os.getenv("DEV_CHANNEL")) if env_dev else int(os.getenv("OUTPUT_CHANEL"))
    outputChannel = bot.get_channel(output_channel_id)

    newPrompt = SynBotPrompt(ctx, outputChannel)
    if newPrompt.isValid:
        await promptToImage(ctx, newPrompt)
    else:
        aiNumInstruction = "2" if syn2 else ""
        await ctx.send(f"{ctx.author.mention} Bad format request. Type **{aiNumInstruction}-helpMe** for instructions")

@bot.command()
async def prompt(ctx):
    # Select proper channel to handle requests
    input_channel_id = int(os.getenv("DEV_CHANNEL")) if env_dev else int(os.getenv("INPUT_CHANEL"))

    # Ignore generate request if it's not coming from the right channel
    inputChannel = bot.get_channel(input_channel_id)
    if ctx.channel != inputChannel:
        print ("Request made in wrong channel")
        return

    # Fetch the channel where the image will be sent to. This might be move, if I ever find a way to thread the API calls.
    output_channel_id = int(os.getenv("DEV_CHANNEL")) if env_dev else int(os.getenv("OUTPUT_CHANEL"))
    outputChannel = bot.get_channel(output_channel_id)

    newPrompt = SynBotPrompt(ctx, outputChannel)
    if not newPrompt.isValid:
        # Required parameters
        if newPrompt.userPrompt == None:
            await ctx.send(f"{ctx.author.mention} -> Missing 'prompt' parameter.")
            return
        elif newPrompt.formatStr == None:
            await ctx.send(f"{ctx.author.mention} -> Missing 'format' parameter.")
            return
        elif newPrompt.formatStr != "landscape" and newPrompt.formatStr != "portrait":
            await ctx.send(f"{ctx.author.mention} -> Parameter 'format' does not have the right value. Use 'landscape' or 'portrait'")
            return
        else:
            await ctx.send(f"{ctx.author.mention} -> Unknown error while parsing message. WTF happened???")
            return
    else:
        await promptToImage(ctx, newPrompt)

async def promptToImage(ctx, newPrompt: SynBotPrompt):

    ################ Create a resume to send back to the author ################################
    # Add hirez in first response to INPUT_CHANNEL
    appendHirez = ""
    if newPrompt.hirez:
        newPrompt.hirezValue = os.getenv("HIREZ_SCALE2") if syn2 else os.getenv("HIREZ_SCALE")
        appendHirez = " **hirez (" + str(newPrompt.hirezValue) + ")**"

    # V1 make character names in BOLD
    resumedPrompt = newPrompt.userPrompt
    for key in charactersLORA.keys():
        if key in newPrompt.userPrompt:
            resumedPrompt = resumedPrompt.replace(key, "**" + key.lower() + "**")

    # V1 make LORAs in BOLD
    for key in LORA_List.keys():
        if key in newPrompt.userPrompt:
            resumedPrompt = resumedPrompt.replace(key, "**" + key.lower() + "**")

    # First response
    # Define the URL this prompt request will use for every API calls
    URL = getAPIURL()
    newPrompt.URL = URL
    synBot = "Syn2-Bot" if syn2 else "Syn-Bot"
    qsizeCount = str(bot.queue.qsize() + 1)
    await ctx.send(f"Queuing request from {ctx.message.author.display_name} , in " + newPrompt.format + appendHirez + " format, on " + synBot + ". (" + qsizeCount + " in queue)")
    ################ END resume ################################

    # Add the prompt to the queue, where it will be executed on next queue loop
    await bot.queue.put(newPrompt)

# Process an image URL and return a base64 encoded string
async def encode_discord_image(image_url):
    try:
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error in encode_discord_image: {e}")
        return None

@bot.command()
async def newOutfit(ctx):

    # await ctx.send(f"{ctx.author.mention.display_name} newOutfit is broken, will be fixed sometime today, I promise.")
    # return


    # Select proper channel to handle requests
    input_channel_id = int(os.getenv("DEV_CHANNEL")) if env_dev else int(os.getenv("INPUT_CHANEL"))

    # Ignore generate request if it's not coming from the right channel
    inputChannel = bot.get_channel(input_channel_id)
    if ctx.channel != inputChannel:
        print ("Request made in wrong channel")
        return

    message = str(ctx.message.content)

    # message should be in format -> !Syn-newOutfit {"character": "sayaka", "outfit": "casual", "denoise": 0.5, "prompt": "my outfit prompt"}
    jsonStr = message.removeprefix("!Syn-newOutfit").removeprefix("!Syn2-newOutfit").strip()
    try:
        jsonData = json.loads(jsonStr)
    except ValueError as e:
        await ctx.send(f"{ctx.author.mention} invalid json: {e}")
        print(f"invalid json: {e} -> {jsonStr}" )
        return
    
    hasAttachments = len(ctx.message.attachments) >= 2

    characterPrompt = ""
    character = jsonData['character'] if "character" in jsonData else None
    if character == None and not hasAttachments:
        await ctx.send(f"{ctx.author.mention} -> Missing 'character' parameter, or image attachments")
        return
    else:

        # This helps (a little) with keeping the original pose
        if character == "sayaka":
            characterPrompt = "<lora:stsayaka:.5>, stSayaka, "
        elif character == "john":
            characterPrompt = "<lora:stJohn:.5>, stJohn, "
        elif character == "allison":
            characterPrompt = "<lora:stAllison2:.5>,  stAllison2, "
        elif character == "katrina":
            characterPrompt = "<lora:stKatrina2:.5>,  stKatrina, "

    width = 728
    height = 728
    if "hirez" in jsonData:
        width = 856
        height = 856

    batch = jsonData["batch"] if "batch" in jsonData else 1
    if batch < 1 or batch > 4:
        batch = 1
    
    # lower rez when batch is not 1 or SD will be too slow. Use the returned seed to hirez the outfit you liked
    if batch > 1:
        width = 512
        height = 512

    # Default pose is A
    pose = jsonData["pose"] if "pose" in jsonData else "A"
    seedToUse = jsonData["seed"] if "seed" in jsonData else -1
    
    # Remove BG default: True
    removeBG = True
    if "removeBG" in jsonData:
        removeBG = jsonData["removeBG"] == "true"

    # Prepare image paths and stop if image is missing
    if not "outfit" in jsonData and not hasAttachments:
        await ctx.send(f"{ctx.author.mention} -> 'outfit' parameter missing.")
        return
    outfit = jsonData["outfit"] if "outfit" in jsonData else None

    # might need to use custom mask, pointless in case images where attached tho
    mask = jsonData["mask"] if "mask" in jsonData else "mask"

    baseImage = ""
    maskImage = ""

    if hasAttachments:
        ###################### START USING USER SUBMITTED BASE AND MASK IMAGE
        print("Loading sent images as attachments...")
        baseImage = await encode_discord_image(ctx.message.attachments[0].url) # Base?
        maskImage = await encode_discord_image(ctx.message.attachments[1].url) # Mask?
        print("Attachments loaded in memory.")
        ###################### END USING USER SUBMITTED BASE AND MASK IMAGE

    else:
        ###################### START DEFAULT BASE AND MASK IMAGE
        baseImagePath = f"./sprites/{character}/{pose}_{outfit}.png"
        if not os.path.exists(baseImagePath):
            await ctx.send(f"{ctx.author.mention} -> {baseImagePath} does not exist.")
            return           
        baseImage = getImageAtPath(baseImagePath)
        
        maskImagePath = f"./sprites/{character}/{pose}_{mask}.png"
        if not os.path.exists(maskImagePath):
            await ctx.send(f"{ctx.author.mention} -> {maskImagePath} does not exist.")
            return
        maskImage = getImageAtPath(maskImagePath)
        ###################### END DEFAULT BASE AND MASK IMAGE

    denoise = jsonData["denoise"]
    prompt = jsonData["prompt"]

    # Replace prompt tags
    for key in charactersLORA.keys():
        if key in prompt:
            print("Found: " + key)
            prompt = prompt.replace(key, charactersLORA[key])

    # Same for LORA Helpers
    for key in LORA_List.keys():
        if key in prompt:
            print("Found: " + key + " in prompt")
            prompt = prompt.replace(key, LORA_List[key])
    

    # API Payload
    payload = {
        "init_images": [ baseImage ], 
        "mask": maskImage, 
        "denoising_strength": denoise, 
        "image_cfg_scale": 7, 
        "mask_blur": 10, 
        "inpaint_full_res": True,                   #choices=["Whole picture", "Only masked"]
        "inpaint_full_res_padding": 32, 
        "inpainting_mask_invert": 0,                #choices=['Inpaint masked', 'Inpaint not masked']
        "sampler_name": "DPM++ 2M Karras", 
        "batch_size": batch, 
        "steps": 30,
        "seed": seedToUse, 
        "cfg_scale": 7, 
        "width": width, "height": height, 
        "prompt": "masterpiece, best_quality, extremely detailed, intricate, high_details, sharp_focus , best_anatomy, hires, (colorful), beautiful, 4k, magical, adorable, (extraordinary:0.6), (((simple_background))), (((white_background))), multiple_views, reference_sheet, " + characterPrompt + prompt, 
        "negative_prompt": "paintings, sketches, fingers, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, (outdoor:1.6), backlight,(ugly:1.331), (duplicate:1.331), (morbid:1.21), (mutilated:1.21), (tranny:1.331), mutated hands, (poorly drawn hands:1.5), blurry, (bad anatomy:1.21), (bad proportions:1.331), extra limbs, (disfigured:1.331), lowers, bad hands, missing fingers, extra digit", 
        "sampler_index": "DPM++ 2M Karras"
    }

    # Add controlNet OpenPose
    payload["alwayson_scripts"] = {
        "controlnet": {
            "args": [
                {
                    "enabled": True,
                    "input_image": baseImage,
                    "module": "openpose",
                    "model": "control_v11p_sd15_openpose [cab727d4]",
                    "weight": .75,  # Apply pose on 75% of the steps
                    "pixel_perfect": True
                },
                {
                    "enabled": True,
                    "input_image": baseImage,
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

    # Define the URL this prompt request will use for every API calls
    URL = getAPIURL()
    synBot = "Syn2-Bot" if syn2 else "Syn-Bot"
    await ctx.send(f"**Creating new outfit** for {character} on **{synBot}** , requested by {ctx.message.author.display_name}.")
    await bot.queue.put(asyncio.create_task(sendPayload(ctx, payload, URL, "sdapi/v1/img2img", f"{ctx.author.mention} generated this outfit for {character}", removeBG)))

async def sendPayload(ctx, payload, URL, apiPath, formattedResponse, removeBG=True, removeControlNetImages=True):
        
        # Sending API call request
        print(f"Sending request to '{URL}' ...")

        async with aiohttp.ClientSession(loop=ctx.bot.loop) as session:
            async with session.post(url=f'{URL}/{apiPath}', json=payload) as response:
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

                        # Skip ControlNet images
                        if removeControlNetImages:
                            batchSize = payload["batch_size"]
                            if r["images"].index(i) >= batchSize: #Last 2 images are always ControlNet, Depth and OpenPose
                                continue # loop to next image

                        # Remove background
                        if removeBG:
                            i = await removeBackground(i, URL, ctx) if removeBG else i

                        # Image is in base64, convert it to a discord.File
                        bytes = io.BytesIO(base64.b64decode(i.split(",",1)[0]))
                        bytes.seek(0)
                        discordFile = discord.File(bytes, filename="{seed}-{ctx.message.author.display_name}.png")
                        discordFiles.append(discordFile)


                    output_channel_id = int(os.getenv("DEV_CHANNEL")) if env_dev else int(os.getenv("OUTPUT_CHANEL"))
                    outputChannel = bot.get_channel(output_channel_id)

                    # Send a response with the image attached
                    await outputChannel.send(formattedResponse + " (seed: " + str(responseSeedUsed) + ")", files=discordFiles)


async def removeBackground(discordFile, URL, ctx):

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

# Run Bot in loop
bot.run(os.getenv("TOKEN"))