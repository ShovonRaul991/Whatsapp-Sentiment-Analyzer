import pandas as pd
from urlextract import URLExtract
from wordcloud import WordCloud
from collections import Counter
import emoji
import Detection_Function


extract = URLExtract()


def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    num_message = df.shape[0]

    words = []
    for message in df['message']:
        words.extend(message.split())

    media_count = df[df['message'] == '<Media omitted>'].shape[0]
    delecetd_message_count = df[df['message'] == 'This message was deleted'].shape[0]

    urls = []
    for message in df['message']:
        urls.extend(extract.find_urls(message))

    return num_message, len(words), len(urls), media_count, delecetd_message_count


def most_busy_user(df):
    x = df['user'].value_counts().head()
    dfx = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percentage'})
    return x, dfx


def create_word_cloud(selected_user, df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>']
    temp = temp[temp['message'] != 'This message was deteted\n']

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    wc = WordCloud(width=1000, height=1000, min_font_size=5, background_color="black")
    df_wc = wc.generate(df['message'].str.cat(sep=" "))
    return df_wc


def most_common_words(selected_user, df):
    
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    temp = temp[temp['message'] != 'This message was deteted\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            # if word not in stop_words:
            words.append(word)

    word_df = pd.DataFrame(Counter(words).most_common(20))

    return word_df


def emoji_analysis(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        x = emoji.distinct_emoji_list(message)
        emojis.extend([c for c in x])
    if(len(emojis)!=0):
        emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
        emoji_df.columns = ['Emoji','Frequency']
        emoji_df.index += 1
        return emoji_df
    else:
        return 0


def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline


def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    dailytimeline = df.groupby('only_date').count()['message'].reset_index()

    return dailytimeline


def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()


def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()


def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap



def message_language_count(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df['language'] = df['message'].apply(lambda x: Detection_Function.Detect_The_lang(x))

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    temp = temp[temp['message'] != 'This message was deteted\n']
    temp['message'] = temp['message'].str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'This is an url\n')
    temp = temp[temp['message'] != 'This is an url\n']

    df_eng = temp[temp['language']=='English']
    df_non_eng = temp[temp['language'] != 'English']
    df_non_eng = df_non_eng.drop(['language'],axis = 1)

    eng_count = df_eng.shape[0]
    non_eng_count = df_non_eng.shape[0]

    return df_eng, df_non_eng, eng_count, non_eng_count


def message_sentiment_count(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df['sentiment'] = df['message'].apply(lambda x: Detection_Function.Detect_The_senti(x))

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    temp = temp[temp['message'] != 'This message was deteted\n']
    
    return temp


def seeSentiment(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    x = df['sentiment'].value_counts().head()
    
    dfx = round((df['sentiment'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'Sentiment', 'sentiment': 'percentage'})
    dfx.index +=1
    return x, dfx

def word_in_emotion(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df_joy = df[df['sentiment']=='joy']
    df_sadness = df[df['sentiment']=='sadness']
    df_fear = df[df['sentiment']=='fear']
    df_anger = df[df['sentiment']=='anger']
    df_surprise = df[df['sentiment']=='surprise']
    df_neutral = df[df['sentiment']=='neutral']
    df_disgust = df[df['sentiment']=='disgust']
    df_shame = df[df['sentiment']=='shame']

    return df_joy,df_sadness,df_fear,df_anger,df_surprise,df_neutral,df_disgust,df_shame
