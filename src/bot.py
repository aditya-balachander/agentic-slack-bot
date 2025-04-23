from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

SLACK_BOT_TOKEN="xoxb-8790763695429-8777895148167-LFqn0WGMCmtzUOCOSdo1EYCC"
SLACK_APP_TOKEN="xapp-1-A08PEMRATUL-8796347222451-6adff9fdad6477e955e945590d5b1944f68d5b1f81293af655d54f462625c51d"

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
    say('Hello World')

if __name__=="__main__":
    handler=SocketModeHandler(app,SLACK_APP_TOKEN)
    handler.start()