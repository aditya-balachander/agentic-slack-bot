from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
import os
load_dotenv()

SLACK_BOT_TOKEN=os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN=os.getenv("SLACK_APP_TOKEN")

app=App(token=SLACK_BOT_TOKEN)
try:
    bot_user_id = app.client.auth_test()['user_id']
    print(f"Detected Bot User ID: {bot_user_id}")
except Exception as e:
    print(f"Error fetching bot user ID: {e}. History reconstruction might be inaccurate.")
    bot_user_id = None # Handle case where auth_test fails

@app.event("app_mention")
def mention_handler(body,say):
    print(body)
    say('Hello! How can I help you?')

if __name__=="__main__":
    handler=SocketModeHandler(app,SLACK_APP_TOKEN)
    handler.start()