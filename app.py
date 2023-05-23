from collections import Counter
import numpy as np
from flask import Flask
from flask import render_template
from flask import request
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import twitter_cred
import tweepy
import smtplib
from email.message import EmailMessage
import mail_cred

app = Flask(__name__)

rslt = -1


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/form_main')
def form_main():
    return render_template('form_main.html')


@app.route('/form')
def form():
    return render_template('form.html')


@app.route('/predict', methods=["POST", "GET"])
def predict():
    global rslt
    if request.method == 'POST':
        screen_name = request.form['screen_name']
        followers_count = int(request.form['followers_count'])
        friends_count = int(request.form['friends_count'])
        listed_count = int(request.form['listed_count'])
        verified = bool(request.form['verified'])
        description = request.form['description']
        statuses_count = int(request.form['statuses_count'])
        name = request.form['name']
        status = request.form['status']
        df = pd.DataFrame(data=[[screen_name, followers_count, friends_count, listed_count, verified, description, statuses_count, name, status]], columns=[
                          "screen_name", "followers_count", "friends_count", "listed_count", "verified", "description", "statuses_count", "name", "status"])
        print(df)
        df['listed_count_binary'] = (df.listed_count > 20000) == False
        bag_of_words_bot = r'bot|prison|paper|follow me|tweet me|swag|bang|b0t|magic|face|wizard|bot|b0t|cannabis|tweet me|mishear|follow me|updates every|gorilla|yes_ofc|forget' \
            r'expos|kill|clit|bbb|butt|fuck|XXX|sex|truthe|fake|anony|free|virus|funky|RNA|kuck|jargon' \
            r'nerd|swag|jack|bang|bonsai|chick|prison|paper|pokem|xx|freak|ffd|dunia|clone|genie|bbb' \
            r'ffd|onlyman|emoji|joke|troll|droop|free|every|wow|cheese|yeah|bio|magic|wizard|face'

        df['screen_name_binary'] = df.screen_name.str.contains(
            bag_of_words_bot, case=False, na=False)
        df['name_binary'] = df.name.str.contains(
            bag_of_words_bot, case=False, na=False)
        df['description_binary'] = df.description.str.contains(
            bag_of_words_bot, case=False, na=False)
        df['status_binary'] = df.status.str.contains(
            bag_of_words_bot, case=False, na=False)
        features = ['screen_name_binary', 'name_binary', 'description_binary', 'status_binary',
                    'verified', 'followers_count', 'friends_count', 'statuses_count', 'listed_count_binary']
        clf = pickle.load(open('model/model_pre.pkl', 'rb'))
        predn = clf.predict(df[features].values)
        print(predn)
        rslt = predn[0]
        return render_template('result.html', rslt=rslt, uname=screen_name)
    else:
        return render_template('index.html')


@app.route('/predict_main', methods=["POST", "GET"])
def predict_main():
    global rslt
    if request.method == 'POST':
        username = request.form['username']

        df = fetch(username)
        # print(df.values)

        if type(df) == type(""):
            return '''
                <script>
                    alert("User Not found!");
                    window.location.href = "/";
                </script>
            '''
        else:
            df2 = df
            df['listed_count_binary'] = (df.listed_count > 20000) == False

            bag_of_words_bot = r'bot|prison|paper|follow me|tweet me|swag|bang|b0t|magic|face|wizard|bot|b0t|cannabis|tweet me|mishear|follow me|updates every|gorilla|yes_ofc|forget' \
                r'expos|kill|clit|bbb|butt|fuck|XXX|sex|truthe|fake|anony|free|virus|funky|RNA|kuck|jargon' \
                r'nerd|swag|jack|bang|bonsai|chick|prison|paper|pokem|xx|freak|ffd|dunia|clone|genie|bbb' \
                r'ffd|onlyman|emoji|joke|troll|droop|free|every|wow|cheese|yeah|bio|magic|wizard|face'

            df['screen_name_binary'] = df.screen_name.str.contains(
                bag_of_words_bot, case=False, na=False)
            df['name_binary'] = df.name.str.contains(
                bag_of_words_bot, case=False, na=False)
            df['description_binary'] = df.description.str.contains(
                bag_of_words_bot, case=False, na=False)
            df['status_binary'] = df.status.str.contains(
                bag_of_words_bot, case=False, na=False)
            features = ['screen_name_binary', 'name_binary', 'description_binary', 'status_binary',
                        'verified', 'followers_count', 'friends_count', 'statuses_count', 'listed_count_binary']
            clf = pickle.load(open('model/model_pre.pkl', 'rb'))
            predn = clf.predict(df[features].values)
            print(predn)
            rslt = predn[0]
            return render_template('result2.html', rslt=rslt, uname=username, df=df2.values[0])
    else:
        return render_template('index.html')


def fetch(username):
    # Set up your Twitter API credentials
    consumer_key = twitter_cred.consumer_key
    consumer_secret = twitter_cred.consumer_secret
    access_token = twitter_cred.access_token
    access_token_secret = twitter_cred.access_token_secret

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)

    try:
        user = api.get_user(screen_name=username)
        if "status" in user._json.keys():
            user_status = user.status._json
        else:
            user_status = ""

        df = pd.DataFrame(data=[[user.screen_name, user.followers_count, user.friends_count, user.listed_count, user.verified, user.description, user.statuses_count, user.name, user_status]], columns=[
            "screen_name", "followers_count", "friends_count", "listed_count", "verified", "description", "statuses_count", "name", "status"])

        return df
    except tweepy.TweepyException as e:
        # print(e)
        if e.response.status_code == 404:
            return "not_found"
        else:
            return "err"


@app.route('/send_mail', methods=['POST', 'GET'])
def send_mail():
    if request.method == 'POST':
        screen_name = request.form['uname']
        sender_email = mail_cred.username
        receiver_email = "support@twitter.com"
        subject = "Bot Account Report"
        body = "Hello,\n This mail is to report that the twitter account with user name '"+screen_name + \
            "' seems to be a bot account. So I request your concern to this issue.\n\nThank You\nTeam TweeBot"

        smtp_server = "smtp.gmail.com"
        smtp_port = 587

        username = mail_cred.username
        password = mail_cred.password

        try:
            server = smtplib.SMTP(smtp_server, smtp_port)

            server.starttls()
            server.login(username, password)

            msg = EmailMessage()
            msg.set_content(body)
            msg["Subject"] = subject
            msg["From"] = sender_email
            msg["To"] = receiver_email

            server.send_message(msg)
            return 's'

        except Exception as e:
            print(e)
            return 'e'
        finally:
            if server:
                server.quit()
    else:
        return render_template('index.html')
