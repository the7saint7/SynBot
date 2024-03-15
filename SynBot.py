import io
import copy
import json
import base64
import discord
import requests
from discord.ext import commands
from charactersList import charactersLORA
from LORA_Helper import LORA_List

TOKEN = "MTIxODAwNzA5MDYyMzI4NzQ1Nw.GFiVd9.p4A95gINQD-AjFL6XT7qczYMQQNKWwEyXoRQVM"
INPUT_CHANEL = 1218052732531773501
OUTPUT_CHANEL = 1218105385895067668
DEV_CHANNEL = 1218006498035105934
HIREZ_SCALE = 1.5

env_dev = False

# Discord Bot
bot = commands.Bot(command_prefix="!Syn-", intents=discord.Intents.all())

@bot.event
async def on_ready():
    channel_id = DEV_CHANNEL if env_dev else INPUT_CHANEL
    channel = bot.get_channel(channel_id)
    await channel.send("Syn AI is online and ready to generate some images. Type **!Syn-helpMe** for instructions!")

@bot.command()
async def helpMe(ctx):
    await ctx.message.author.send("You can ask me to generate ST-related images using these keywords: !Syn-generate : FORMAT[landscape|portrait]: prompt with names like JOHN, KATRINA, in UPPERCASE\n**!Syn-generate : landscape: JOHN as a girl at a beach, wearing a bikini**\n**!Syn-generate : portrait: KATRINA wearing a school uniform in a classroom yelling at someone else**\nYou can get a full list of supported ST character names by using the command **!Syn-characters**\nYou can get a list of advanced parameters by calling **!Syn-advanced**\nYou can get a list pre-formated LORAs by calling **!Syn-loras**")

@bot.command()
async def advanced(ctx):
    await ctx.message.author.send("Here is a list of extra parameters you can add at the end of your prompt. Make sure you add an extra ':' to seperate your prompt from the extra parameters.\n**hirex=true**: Will increase the resolution of the image and make it look sharper. This feature could be disabled if abused as it taxes my PC a lot.\n**seed=654789**: You can enter any seed NUMBER yo want. This way you can change your prompt a little and use the same seed, to see how it changes.\nA neat trick, is to NOT use hirez, until you find an image that you like, then you can use the same prompt, pass it the hirez and seed number and it will generate the same image but in higher quality & resolution.\n\nExample:\n**!Syn-generate : portrait: JOHN, TKUNIFORM, a girl wearing a school uniform in a classroom angry, frown, hands_on_own_hips, hand_on_hip, large_breasts, long_hair, ponytail : hirez=true, seed=3955923732**")

@bot.command()
async def characters(ctx):
    characterNames = "\n".join(charactersLORA.keys())
    await ctx.message.author.send(f"Here's a list of ST character names you can use in your prompt. Remeber to write them in uppercase.\n{characterNames}")

@bot.command()
async def loras(ctx):
    list = "\n".join(LORA_List.keys())
    await ctx.message.author.send(f"Here's a list of extra tags that will be converted to some lora preset I've compiled for you.\n{list}")

@bot.command()
async def generate(ctx):

    # Select proper channel to handle requests
    channel_id = DEV_CHANNEL if env_dev else INPUT_CHANEL

    # Ignore generate request if it's not coming from the right channel
    inputChannel = bot.get_channel(channel_id)
    if ctx.channel != inputChannel:
        print ("Request made in wrong channel")
        return

    # The message that was sent
    message = str(ctx.message.content)

    # Split the message with our delimiter - :
    # !Syn-generate : FORMAT : PROMPT : EXTRA-PARAMETERS
    data = message.split(":")
    if len(data) >= 3:
        # Ignore data[0]
        format = "640x360" if data[1].strip() == "landscape" else "360x640"
        userPrompt = data[2]
        parameters = data[3].split(",") if len(data) > 3 else []

        # Check for parameters, if any
        hirez = False
        seedToUse = -1
        for paramData in parameters:
            param = paramData.split("=")
            if param[0].strip() == "hirez" and param[1].strip() == "true":
                hirez = True
            if param[0].strip() == "seed":
                seedToUse = param[1].strip()
                

        # Add hirez in first response to INPUT_CHANNEL
        appendHirez = ""
        if hirez:
            appendHirez = " **hirez (" + str(HIREZ_SCALE) + ")**"


        # V1 make character names in BOLD
        resumedPrompt = userPrompt
        for key in charactersLORA.keys():
            if key in userPrompt:
                resumedPrompt = resumedPrompt.replace(key, "**" + key.lower() + "**")

        # V1 make LORAs in BOLD
        for key in LORA_List.keys():
            if key in userPrompt:
                resumedPrompt = resumedPrompt.replace(key, "**" + key.lower() + "**")

        # First response
        await inputChannel.send(f"Queuing request from {ctx.message.author} , in " + format + appendHirez + " format")

        # The generate image call and callback
        await generateImage(ctx, userPrompt, format, hirez, seedToUse)
    else:
        await ctx.send(f"{ctx.author.mention} Bad format request. Type **!Syn-helpMe** for instructions")

async def generateImage(ctx, userPrompt, format, hirez=False, seedToUse=-1):

    # fix prompt by replacing character names with their LORAs
    fixedPrompt = userPrompt
    for key in charactersLORA.keys():
        if key in userPrompt:
            print("Found: " + key)
            fixedPrompt = fixedPrompt.replace(key, charactersLORA[key])

    # Same for LORA Helpers
    for key in LORA_List.keys():
        if key in userPrompt:
            print("Found: " + key)
            fixedPrompt = fixedPrompt.replace(key, LORA_List[key])

    payload = {
        "prompt": "masterpiece, best_quality, extremely detailed, intricate, high_details, sharp_focus , best_anatomy, hires, (colorful), beautiful, 4k, magical, adorable, (extraordinary:0.6), <lora:thickline_fp16:.2>, <lora:neg4all_bdsqlsz_V3.5:1.0>, negative_hand-neg, " + fixedPrompt,
        "negative_prompt": "head out of frame, fewer digits, extra body parts, censored, collage, logo, border, badhandv4, paintings, sketches, fingers, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, (outdoor:1.6), backlight,(ugly:1.331), (duplicate:1.331), (morbid:1.21), (mutilated:1.21), (tranny:1.331), mutated hands, (poorly drawn hands:1.5), blurry, (bad anatomy:1.21), (bad proportions:1.331), extra limbs, (disfigured:1.331), lowers, bad hands, missing fingers, extra digit",
        "sampler_name": "DPM++ 2M Karras",
        "batch_size": 1,
        "steps": 50,
        "cfg_scale": 7,
        "width": int(format.split("x")[0]),
        "height": int(format.split("x")[1]),
        "restore_faces": False,
        "seed": seedToUse
    }

    if hirez:
        payload["denoising_strength"] = 0.5
        payload["enable_hr"] = True
        payload["hr_upscaler"] = "4x-UltraSharp"
        #payload["hr_resize_x"] = int(format.split("x")[0]) * HIREZ_SCALE
        #payload["hr_resize_y"] = int(format.split("x")[1]) * HIREZ_SCALE
        payload["hr_scale"] = HIREZ_SCALE
        payload["hr_sampler_name"] = "DPM++ 2M Karras"
        payload["hr_second_pass_steps"] = 20


    url = "http://localhost:7860"

    # Sending API call request
    print("Sending request...")
    try:
        response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)
    print("Request returned: " + str(response.status_code))

    # Making sure we respond in the right channel
    channel_id = DEV_CHANNEL if env_dev else OUTPUT_CHANEL
    channel = bot.get_channel(channel_id)

    # Convert response to json
    r=response.json()   

    # Extract the seed that was used to generate the image
    info = r["info"]
    infoJson = json.loads(info)
    seedUsed = infoJson["seed"]

    # looping response to get actual image
    for i in r['images']:
        # Image is in base64, convert it to a discord.File
        bytes = io.BytesIO(base64.b64decode(i.split(",",1)[0]))
        bytes.seek(0)
        discordFile = discord.File(bytes, filename="{seed}-{ctx.message.author}.png")

        # Send a response with the image attached
        await channel.send(f"{ctx.author.mention} generated this image with prompt:{ctx.message.jump_url} and seed: {seedUsed}", file=discordFile)


# Run Bot in loop
bot.run(TOKEN)