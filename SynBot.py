import io
import sys
import json
import base64
import asyncio
import discord
import requests
from discord.ext import commands
from charactersList import charactersLORA
from LORA_Helper import LORA_List
from openPoses import getPose, getLewdPose, getImageAtPath
from SynBotMain import SynBotManager, SynBotPrompt
import os
from dotenv import load_dotenv

load_dotenv()
env_dev = False
if len(sys.argv) > 1:
    env_dev = True if sys.argv[1] == "dev" else False

# Discord Bot
bot = SynBotManager(command_prefix="!Syn-", intents=discord.Intents.all())

@bot.event
async def on_ready():
    channel_id = int(os.getenv("DEV_CHANNEL")) if env_dev else int(os.getenv("INPUT_CHANEL"))
    channel = bot.get_channel(channel_id)
    await channel.send("Syn AI is online and ready to generate some images. Type **!Syn-helpMe** for instructions!")

@bot.command()
async def helpMe(ctx):
    await ctx.message.author.send("You can ask me to generate ST-related images using these keywords: !Syn-generate : FORMAT[landscape|portrait]: prompt with names like JOHN, KATRINA, in UPPERCASE\n**!Syn-generate : landscape: JOHN as a girl at a beach, wearing a bikini**\n**!Syn-generate : portrait: KATRINA wearing a school uniform in a classroom yelling at someone else**\nYou can get a full list of supported ST character names by using the command **!Syn-characters**\nYou can get a list of advanced parameters by calling **!Syn-advanced**\nYou can get a list pre-formated LORAs by calling **!Syn-loras**\nYou can also check the list of available poses in the **syn-ai-help** forum")

@bot.command()
async def advanced(ctx):
    await ctx.message.author.send("Here is a list of extra parameters you can add at the end of your prompt. Make sure you add an extra ':' to separate your prompt from the extra parameters.\n**hirez=true**: Will increase the resolution of the image and make it look sharper. This feature could be disabled if abused as it taxes my PC a lot.\n**removeBG=true**: Will try to remove the background from the image. Results will vary.\n**seed=654789**: You can enter any seed NUMBER yo want. This way you can change your prompt a little and use the same seed, to see how it changes.\n**batch=4**: will create 4 images, you can use any number between 1 and 4. batch count will be ignore if hirez is True.\nA neat trick, is to NOT use hirez, until you find an image that you like, then you can use the same prompt, pass it the hirez and seed number and it will generate the same image but in higher quality & resolution.\n\nExample:\n**!Syn-generate : portrait: JOHN, TKUNIFORM, a girl wearing a school uniform in a classroom angry, frown, hands_on_own_hips, hand_on_hip, large_breasts, long_hair, ponytail : hirez=true, seed=3955923732**")

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
        await ctx.send(f"{ctx.author.mention} Bad format request. Type **!Syn-helpMe** for instructions")

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
        appendHirez = " **hirez (" + str(os.getenv("HIREZ_SCALE")) + ")**"

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
    qsizeCount = str(bot.queue.qsize() + 1)
    await ctx.send(f"Queuing request from {ctx.message.author} , in " + newPrompt.format + appendHirez + " format. (" + qsizeCount + " in queue)")
    ################ END resume ################################

    # Add the prompt to the queue, where it will be executed on next queue loop
    await bot.queue.put(newPrompt)



@bot.command()
async def newOutfit(ctx):
    # Select proper channel to handle requests
    input_channel_id = int(os.getenv("DEV_CHANNEL")) if env_dev else int(os.getenv("INPUT_CHANEL"))

    # Ignore generate request if it's not coming from the right channel
    inputChannel = bot.get_channel(input_channel_id)
    if ctx.channel != inputChannel:
        print ("Request made in wrong channel")
        return

    message = str(ctx.message.content)

    # message should be in format -> !Syn-newOutfit {"character": "sayaka", "outfit": "casual", "denoise": 0.5, "prompt": "my outfit prompt"}
    jsonStr = message.removeprefix("!Syn-newOutfit").strip()
    jsonData = json.loads(jsonStr)

    characterPrompt = ""
    character = jsonData['character']
    if character == None:
        await ctx.send(f"{ctx.author.mention} -> Missing 'character' parameter.")
        return
    else:

        # This helps (a little) with keeping the original pose
        if character == "sayaka":
            characterPrompt = "<lora:stsayaka:.5>, stSayaka, "
        elif character == "john":
            characterPrompt = "<lora:stJohn:.5>, stJohn, "
        elif character == "allison":
            characterPrompt = "<lora:stAllison2:.5>,  stAllison2, "

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
    
    seedToUse = jsonData["seed"] if "seed" in jsonData else -1
    removeBG = True if "removeBG" in jsonData else False

    # Prepare image paths and stop if image is missing
    if not "outfit" in jsonData:
        await ctx.send(f"{ctx.author.mention} -> 'outfit' parameter missing.")
        return
    if not "character" in jsonData:
        await ctx.send(f"{ctx.author.mention} -> 'character' parameter missing.")
        return
    outfit = jsonData["outfit"]

    # might need to use custom mask
    mask = jsonData["mask"] if "mask" in jsonData else "mask"

    #Lets help the users by making sure the right mask image is being used in special cases
    if character == "allison" and outfit == "gym":
        mask = "gym_mask"


    baseImagePath = f"./sprites/{character}/{outfit}.png"
    if not os.path.exists(baseImagePath):
        await ctx.send(f"{ctx.author.mention} -> {baseImagePath} does not exist.")
        return           
    
    maskImagePath = f"./sprites/{character}/{mask}.png"
    if not os.path.exists(maskImagePath):
        await ctx.send(f"{ctx.author.mention} -> {maskImagePath} does not exist.")
        return
    
    denoise = jsonData["denoise"]
    prompt = jsonData["prompt"]

    baseImage = getImageAtPath(baseImagePath)
    # API Payload
    payload = {
        "init_images": [ baseImage ], 
        "mask": getImageAtPath(maskImagePath), 
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
    await ctx.send(f"Creating new outfit for {character}, requested by {ctx.message.author}.")
    await bot.queue.put(asyncio.create_task(sendPayload(ctx, payload, "sdapi/v1/img2img", f"{ctx.author.mention} generated this outfit for {character}")))

async def sendPayload(ctx, payload, apiPath, formattedResponse, removeBG=True, removeControlNetImages=True):
        
        # Sending API call request
        print("Sending request...")
        try:
            baseURL = os.getenv("SD_API_URL")
            response = requests.post(url=f'{baseURL}/{apiPath}', json=payload)
            response.raise_for_status()
        except requests.exceptions.HTTPError as err:
            raise SystemExit(err)
        print("Request returned: " + str(response.status_code))

        # Convert response to json
        r = response.json()

        # Extract the seed that was used to generate the image
        info = r["info"]
        infoJson = json.loads(info)
        responseSeedUsed = infoJson["seed"]

        # looping response to get actual image
        discordFiles = []
        for i in r['images']:

            # Skip ControlNet images
            if removeControlNetImages:
                if r["images"].index(i) >= len(r["images"]) -2:
                    pass

            # Remove background
            i = await removeBackground(i) if removeBG else i

            # Image is in base64, convert it to a discord.File
            bytes = io.BytesIO(base64.b64decode(i.split(",",1)[0]))
            bytes.seek(0)
            discordFile = discord.File(bytes, filename="{seed}-{ctx.message.author}.png")
            discordFiles.append(discordFile)


        # if removeBG:
        #     await removeBackground(discordFiles)


        output_channel_id = int(os.getenv("DEV_CHANNEL")) if env_dev else int(os.getenv("OUTPUT_CHANEL"))
        outputChannel = bot.get_channel(output_channel_id)

        # Send a response with the image attached
        await outputChannel.send(formattedResponse + " (seed: " + str(responseSeedUsed) + ")", files=discordFiles)


async def removeBackground(discordFile):

    # for discordFile in discordFiles:
    payload = {
        "input_image": discordFile,
        "model": "isnet-anime"
    }

    # Sending API call request
    baseURL = os.getenv("SD_API_URL")
    print("Sending bg remove request...")
    try:
        response = requests.post(url=f'{baseURL}/rembg', json=payload)
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)
    print("BG remove request returned: " + str(response.status_code))

    # Convert response to json
    r=response.json()
    return r['image']
    # if i != None:
    #     # Image is in base64, convert it to a discord.File
    #     bytes = io.BytesIO(base64.b64decode(i))
    #     bytes.seek(0)
    #     newDiscordFile = discord.File(bytes, filename="{seed}-{ctx.message.author}_transparent.png")
    #     discordFiles.insert(0, newDiscordFile)

# Run Bot in loop
bot.run(os.getenv("TOKEN"))