This folder contains the Docker code for deploying a secure container
containing an Openclaw agent

1. Make a directory to persist openclaw data as a volume
    mkdir ~/.openclaw
2. Copy an existing config file into the volume
    cp openclaw/openclaw.json  ~/.openclaw/
3. Run the container 
        -it keeps std open and gives terminal 
        --rm removes the continer when you exit 
        -e sets environment variables 
            *need to figure out model for API key
        --cap-drop removes all capabilities
        mount the current directory in the container
        mount the config file in the container 
        set workspace
    docker run -it --rm \
        --name openclaw \
        -e OPENCLAW_GATEWAY_TOKEN=${OPENCLAW_GATEWAY_TOKEN} \
        -e GEMINI_API_KEY=${GEMINI_API_KEY} \
        --cap-drop ALL \
        -v $PWD:/work \
        -v ~/.openclaw:/home/openclaw/.openclaw \
        -w /work \



 






Notes:

The idea is to set up an agent on a machine that controls arms 
and take user input to perform a variety of tasks 

Requires instalation and onboarding
    install OC on the machine   
    onboarding creates a config file

Dependencies 
    Node 
    API key / local model

Onboarding Configures
    Model / auth
        provider name 
        API key
    Workspace
    Gateway
    Channels
        how to pass messages to OC 
        can set up with Discord etc.. 
            probably not relevant - will be running in person from terminal
            but is an option 
    Deamon

Hooks 
    small scripts that run inside gateway when events fire 

Gateway 
    runs as a Daemon (process) in container 
    runs a control UI over a port
    serves as the entrypoint to the container with OC inside 




Arch
    Conatainer
        contains OC with gateway(daemon)
        exposes control UI
        configures channel 
        configures model 
    Config Volume 
        persists openclaw.json
    Environment Variables
        holds secrets like API keys
    Data Volume 
        persists openclaw memory and state
    








    