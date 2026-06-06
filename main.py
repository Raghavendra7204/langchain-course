from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv()


#llm = ChatGoogleGenerativeAI(temperature=0,model="gemini-2.5-flash")
llm = ChatOllama(model="gemma3:270m", temperature=0)

information = '''Wesley Wales Anderson (born May 1, 1969) is an American filmmaker. His films are known for themes of grief, loss of innocence, and dysfunctional families. Due to his films' eccentricity, distinctive visual and narrative styles, and frequent use of ensemble casts, critics have described Anderson as an auteur. He has received various accolades, including an Academy Award, a BAFTA Award, and nominations for four Golden Globe Awards. Three of his films appeared in BBC Culture's 2016 poll of the greatest films since 2000.

Anderson gained acclaim for his early films Bottle Rocket (1996) and Rushmore (1998). He often collaborated with the brothers Luke Wilson and Owen Wilson during that time and founded his production company American Empirical Pictures. He received a nomination for the Academy Award for Best Original Screenplay for The Royal Tenenbaums (2001). His next films included The Life Aquatic with Steve Zissou (2004), The Darjeeling Limited (2007), and his first stop-motion film, Fantastic Mr. Fox (2009), for which he received a Best Animated Feature nomination, and then Moonrise Kingdom (2012), earning his second Best Original Screenplay nomination.

For his film The Grand Budapest Hotel (2014), he received his first Academy Award nominations for Best Director and Best Picture, and also his third Best Original Screenplay nomination, and won the BAFTA Award for Best Original Screenplay. Later works include his second stop-motion film, Isle of Dogs (2018), earning him the Silver Bear for Best Director and another Best Animated Feature nomination,[1] followed by The French Dispatch (2021), Asteroid City (2023) and The Phoenician Scheme (2025). Anderson won the Academy Award for Best Live Action Short Film for The Wonderful Story of Henry Sugar (2023). Rushmore and The Grand Budapest Hotel have been inducted into the National Film Registry.'''


summary_template = '''Given the following information {information} about a person, I want you to : 
1. Summarize the information in a concise manner.
2. Give me 3 interesting facts about the person.
3. What is the person's most notable work?'''

summary_prompt_template = PromptTemplate(template=summary_template, input_variables=["information"])

chain = summary_prompt_template | llm
respose=chain.invoke(input={"information": information})
print(respose.content)


