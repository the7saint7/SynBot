# Requirements
- A Stable Diffusion Automatic 1111 running with the '--api' parameter, to enable calling the API
    - Install "ControlNet" extension
        - Download and install ControlNet models:
            - [OpenPose](https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth?download=true)
            - [Depth](https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth?download=true_)
            - [SoftEdge](https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_softedge.pth?download=true)
    - Install "rembg" extension
    - Install "4x-UltraSharp" upscaler [Find it here](https://openmodeldb.info/models/4x-UltraSharp) put the file in your SD folder (SD/models/ESRGAN/4x-UltraSharp.pth)
- A Discord server where you are Admin



# Enabling your own Bot and setting up your Discord server
You can find many tutorials on the web on how to setup your own Discord Bot (I gave my Bot admin-role, you may want to be more careful)
Heres my recommended link: [Create a Discord Bot](https://discordpy.readthedocs.io/en/stable/discord.html)

- Put the generated TOKEN in the .env file in src/
- Create a FORUM channel, where the Bot can publish the generated images
    - The Forum must have a few tags or the code might crash. Its up to you to remove the taging code, or to add the tags in the forum.
        - txt2img
        - img2img
        - inpaint
        - outfits
        - birth
        - expressions
        - removeBG
        - superHiRez
- Only allow Admins(or what ever role you gave your Bot) to publish messages on the FORUM (optional)
- Enable developer mode in your discord app, to be able to get your channelIDs
    - [Enable developer mode and get channelID](https://www.howtogeek.com/714348/how-to-enable-or-disable-developer-mode-on-discord/#:~:text=In%20Discord's%20settings%20menu%2C%20select,the%20%22Developer%20Mode%22%20option.)
    - Put the channelID for the FORUM in the "OUTPUT_CHANEL" of the .env file
    - Either create or select an already existing text channel and get it's channelID the same way. Save the ID in the "INPUT_CHANEL" variable of the .env file 


# Installing up python requirement
Navigate a terminal to the "src" folder and run the command:
```
pip install -r requirements.txt
```
This will install discord.py and many other libraries I used to build Syn-Bot

# Running the Bot
To lunch your own Syn-Bot, navigate inside the src folder and run the command:
```
py SynBot.py
```

To run Syn2, type:
```
py SynBot.py syn2
```

To run dev environment (remember to setup a proper "DEV_CHANNEL" and "DEV_FORUM")
```
py SynBot.py dev
```

That should be all!