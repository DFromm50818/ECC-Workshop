import json
import asyncio
import semantic_kernel as sk
from services import Service
from openai import AzureOpenAI
from dotenv import dotenv_values

class RecommendationEngine:    
    def __init__(self):
        config = dotenv_values(".env")

        # uses the USE_AZURE_OPENAI variable from the .env file to determine which AI service to use
        # false means use OpenAI, True means use Azure OpenAI
        selectedService = Service.AzureOpenAI if config.get("USE_AZURE_OPENAI") == "True" else Service.OpenAI

        if selectedService == Service.AzureOpenAI:
            deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
            self.deployment = deployment
            self.client = AzureOpenAI(azure_endpoint = endpoint, 
                        api_key=api_key,  
                        api_version="2024-02-15-preview"
                        )
        else:
            raise Exception("OpenAI not implemented")    