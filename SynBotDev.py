import os
# import io
import sys
# import json
# import base64
# import aiohttp
# import asyncio
import discord
# import requests
from PIL import Image
from dotenv import load_dotenv
from discord.ext import commands
from LORA_Helper import LORA_List
from charactersList import charactersLORA
from SynBotMain import SynBotManager, SynBotPrompt
# from openPoses import getPose, getLewdPose, getImageAtPath

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

# Each "-2IMG" should be calling this at the end of their "parse and prepare data" phase
async def addToQueue(ctx, newPrompt: SynBotPrompt):

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
    newPrompt.URL = getAPIURL()
    synBot = "Syn2-Bot" if syn2 else "Syn-Bot"
    if newPrompt.type == "txt2img":
        await ctx.send(f"Queuing request from {ctx.message.author.display_name} , in " + newPrompt.format + appendHirez + " format, on " + synBot + ".")
    elif newPrompt.type == "outfits":
        await ctx.send(f"**Creating new outfit** for {newPrompt.outfitsCharacter} on **{synBot}** , requested by {ctx.message.author.display_name}.")
    else:
        await ctx.send(f"Queuing request from {ctx.message.author.display_name}, on " + synBot + ".")
    ################ END resume ################################

    # Add the prompt to the queue, where it will be executed on next queue loop
    await bot.queue.put(newPrompt)

# TXT2IMG
@bot.command()
async def txt2img(ctx):
    # Select proper channel to handle requests
    input_channel_id = int(os.getenv("DEV_CHANNEL")) if env_dev else int(os.getenv("INPUT_CHANEL"))

    # Ignore generated request if it's not coming from the right channel
    inputChannel = bot.get_channel(input_channel_id)
    if ctx.channel != inputChannel:
        print ("Request made in wrong channel")
        return

    # Fetch the channel where the image will be sent to. This might be move, if I ever find a way to thread the API calls.
    output_channel_id = int(os.getenv("DEV_FORUM")) if env_dev else int(os.getenv("FORUM_CHANNEL"))
    outputChannel = bot.get_channel(output_channel_id)

    newPrompt = SynBotPrompt(ctx, outputChannel, type="txt2img")
    if newPrompt.isValid:
        await addToQueue(ctx, newPrompt)
    else:
        await ctx.send(newPrompt.errorMsg)        

# IMG2IMG
@bot.command()
async def img2img(ctx):
    # Select proper channel to handle requests
    input_channel_id = int(os.getenv("DEV_CHANNEL")) if env_dev else int(os.getenv("INPUT_CHANEL"))

    # Ignore generated request if it's not coming from the right channel
    inputChannel = bot.get_channel(input_channel_id)
    if ctx.channel != inputChannel:
        print ("Request made in wrong channel")
        return

    # Fetch the channel where the image will be sent to. This might be move, if I ever find a way to thread the API calls.
    output_channel_id = int(os.getenv("DEV_FORUM")) if env_dev else int(os.getenv("FORUM_CHANNEL"))
    outputChannel = bot.get_channel(output_channel_id)

    newPrompt = SynBotPrompt(ctx, outputChannel, type="img2img")
    if newPrompt.isValid:
        await addToQueue(ctx, newPrompt)
    else:
        await ctx.send(newPrompt.errorMsg)

@bot.command()
async def inpaint(ctx):
    # Select proper channel to handle requests
    input_channel_id = int(os.getenv("DEV_CHANNEL")) if env_dev else int(os.getenv("INPUT_CHANEL"))

    # Ignore generated request if it's not coming from the right channel
    inputChannel = bot.get_channel(input_channel_id)
    if ctx.channel != inputChannel:
        print ("Request made in wrong channel")
        return

    # Fetch the channel where the image will be sent to. This might be move, if I ever find a way to thread the API calls.
    output_channel_id = int(os.getenv("DEV_FORUM")) if env_dev else int(os.getenv("FORUM_CHANNEL"))
    outputChannel = bot.get_channel(output_channel_id)

    newPrompt = SynBotPrompt(ctx, outputChannel, type="inpaint")
    if newPrompt.isValid:
        await addToQueue(ctx, newPrompt)
    else:
        await ctx.send(newPrompt.errorMsg)        

@bot.command()
async def outfits(ctx):
    # Select proper channel to handle requests
    input_channel_id = int(os.getenv("DEV_CHANNEL")) if env_dev else int(os.getenv("INPUT_CHANEL"))

    # Ignore generated request if it's not coming from the right channel
    inputChannel = bot.get_channel(input_channel_id)
    if ctx.channel != inputChannel:
        print ("Request made in wrong channel")
        return

    # Fetch the channel where the image will be sent to. This might be move, if I ever find a way to thread the API calls.
    output_channel_id = int(os.getenv("DEV_FORUM")) if env_dev else int(os.getenv("FORUM_CHANNEL"))
    outputChannel = bot.get_channel(output_channel_id)

    newPrompt = SynBotPrompt(ctx, outputChannel, type="outfits")
    if newPrompt.isValid:
        await addToQueue(ctx, newPrompt)
    else:
        await ctx.send(newPrompt.errorMsg)        

# Run Bot in loop
bot.run(os.getenv("TOKEN"))