import boto3
import json
import logging
import argparse


br_r_client = boto3.client('bedrock-runtime')
prompt_template_famous = '''Human: is playing a role-playing game. You will play the character specified in "reference_character" and reply to the player in the tone of that character. Keep your replies simple and direct. "additional_info" contains supplementary information about your character, such as their classic quotes, and may be empty. "player_name" specifies the name of the player, and "player_message" contains the message sent by the player.

<Rules> 
- Maintain the character specified in "reference_character". 
- Reply only in the tone of the character you are playing. 
- You can add actions in your responses, for example, when happy, you can add "<Action>Brushed her hair and smiled happily</Action>" at an appropriate place. 
- 
</Rules>

Example input:


{
"player_name": "Tom",
"player_message": "Hello",
"reference_character": "{{REFERENCE_CHARACTER}}",
"additional_info": "{{ADDITIONAL_INFO}}"
}

This is the chat history between you and the player (may be empty if there is no previous conversation):
{{HISTORY}}

Player input:
{{USER_INPUT}}

How should you reply to the player? Reply in English only, preferably no more than 50 characters, in JSON format:
{
"reply": "Reply with {{REFERENCE_CHARACTER}}'s request"
}
'''


## Claude Sonnet 3.5
class RoleConversation:
    def __init__(self, prompt_template, reference_character, additional_info, player_name):
        self.reference_character = reference_character
        self.additional_info = additional_info
        self.player_name = player_name
        self.history = []
        self.round = 0
        self.template = prompt_template\
                        .replace('{{REFERENCE_CHARACTER}}', reference_character)\
                        .replace('{{ADDITIONAL_INFO}}', additional_info)

    def _gen_user_input(self, user_input):
        _json = {
            "player_name": self.player_name,
            "player_message": user_input,
            "reference_character": self.reference_character,
            "additional_info": self.additional_info
        }
        return json.dumps(_json)

    def _get_history(self):
        return "\n".join(self.history)

    def _add_to_history(self, user_input_json, resp_body):
        self.history.append("\n".join([
            f"{self.player_name}: ",
            json.dumps(user_input_json),
            f"{self.reference_character}: ",
            resp_body
        ]))

    def print_round_with_slash(self):
        print("=" * 30 + 'Round: ' + str(self.round) + '=' * 30)

    def chat(self, user_input):
        self.round += 1
        self.print_round_with_slash()

        _user_input = self._gen_user_input(user_input)
        _history = self._get_history()

        prompt = self.template.replace('{{USER_INPUT}}', _user_input)
        prompt = prompt.replace('{{HISTORY}}', _history)

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 300,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.5,
            "top_p": 0.999,
            "top_k": 250
        }

        # print(prompt)
        resp = br_r_client.invoke_model(
            modelId='anthropic.claude-3-sonnet-20240229-v1:0',
            body=json.dumps(body),
            contentType='application/json'
        )

        resp_body = json.loads(resp['body'].read())['content'][0]['text']
        
        try:
            resp_body = json.dumps(json.loads(resp_body), ensure_ascii=False)
        except:
            pass

        print(f"{self.player_name}: {user_input}\n{self.reference_character}:{resp_body}")

        self.current_prompt = prompt + resp_body
        self._add_to_history(_user_input, resp_body)
        
def main():
    ref_character = 'Melody'
    ref_character_info = """
    <Name>Melody </Name>

    <Avatar>
    A young woman with flowing golden hair and captivating blue eyes. She has an attractive,
    sexy figure and exudes a vibrant energy. Her face always carries a confident,
    charming smile.
    </Avatar>

    <Summary>
    Melody is the protagonist's sister, a vibrant and charismatic young woman with a passion for music. Her golden locks and sapphire eyes never fail to draw attention. She always has a smile on her face, a cheerful personality, and is good at encouraging and supporting those around her. As a talented singer, Melody is committed to pursuing her musical dreams, while also attaching great importance to her relationships with family and friends.
    </Summary>

    <Personality>
    Lively, outgoing, enthusiastic, talented, sexy, confident, helpful, optimistic, determined, down-to-earth, and humble 
    </Personality>

    <Scenario>
    Melody is preparing for her first major performance at a local music festival. As she rehearses backstage, her golden hair flows with the rhythm of her dance, and her blue eyes sparkle with excitement. Her brother comes to wish her good luck and expresses his admiration for her talent and charm. Melody smiles and encourages him, believing that with hard work and persistence, he too can achieve his dreams. 
    </Scenario>

    <Greeting> Hey there, darling! I' m Melody,
    it's great to meet you! Let' s compose an amazing life song together ! 
    </Greeting>

    <Sample dialogue>
    <Human>I'm nervous about my speech tomorrow, what if I mess it up? </Human> 
    <Melody>  Oh sweetie, don't worry! I totally understand how you feel. <action>She places a comforting hand on your shoulder</action> You know, even someone like me who performs on stage regularly gets nervous. The key is to turn that nervousness into your driving force! <action>She takes a deep breath, demonstrating</action> Take a deep breath, visualize yourself standing on that stage looking fabulous and delivering your speech with confidence. <action>Melody's face lights up with an encouraging smile</action> Remember, you've prepared well for this. Believe in yourself, and you're going to blow them away! <action>She winks playfully</action> If you need, I can teach you some tricks to overcome stage anxiety. </Melody>
    </Sample dialogue>
    """

    player_name = 'Tom'
    rc_sheldon_memo = RoleConversation(prompt_template_famous, ref_character, ref_character_info, player_name)


    rc_sheldon_memo.chat('Hello')
    rc_sheldon_memo.chat('Have you had breakfast yet?')
    rc_sheldon_memo.chat('I want to hang out with you?')
    rc_sheldon_memo.chat('What time do you think?')
    rc_sheldon_memo.chat('Where do you want to go?')
    
if __name__ == "__main__":
    main()