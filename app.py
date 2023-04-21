import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

import helper
import preprocessor
import Detection_Function


st.set_page_config(page_title="WHATSAPP CHAT ANALYZER AND PREDICTION APPLICATION",page_icon='a2zapplogo.png',layout="wide")
page_bg_image = """
<style>
[data-testid = "stSidebar"]{
background-image : url("https://w0.peakpx.com/wallpaper/492/145/HD-wallpaper-whatsapp-conversa-fundo-logo.jpg");
background-size: cover;
width: 60%;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid = "stAppViewContainer"]{
    background-image: url("https://w0.peakpx.com/wallpaper/946/21/HD-wallpaper-whatsapp-theme-background-green-original-simple-texture.jpg");
    background-repeat: no-repeat;
    background-size: cover;
    
}
[data-testid="stMarkdownContainer"]{
    color:white
}
[data-testid = "stFileUplodDropzone"]{
background-color: black
}
</style>
"""
st.markdown(page_bg_image,unsafe_allow_html=True)
st.sidebar.markdown(f'<h1 style="color:#33ff33;font-size:24px;">{"WhatsApp Chat Analyzer and Prediction Application"}</h1>',unsafe_allow_html=True)

timeFormat = st.sidebar.radio(
    "What\'s your chat's time format:",
    ('12h', '24h'))
uploaded_file = st.sidebar.file_uploader("Choose WhatsApp Export file",type=['txt'],)
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.pre_process(data,timeFormat)
    df.index += 1

    
    # fetch unique user
    if len(df.axes[0]!=0):
        st.title("Created DataFrame is: ")
        st.dataframe(df)
        user_list = df['user'].unique().tolist()
        user_list.remove('group_notification')
        user_list.sort()
        user_list.insert(0, "Overall")
        
        selected_user = st.sidebar.selectbox("Show analysis with respect to: ", user_list)

        if st.sidebar.button("Show Analysis"):

            num_messages, no_words, no_urls, media_count, deleted_count = helper.fetch_stats(selected_user, df)
            col1, col2, col3 = st.columns(3)

            with col1:
                st.header("Total Messages")
                st.title(num_messages)

            with col2:
                st.header("Total Words")
                st.title(no_words)

            with col3:
                st.header("Total URLs")
                st.title(no_urls)
            

            # monthly timeline
            st.title("Monthly Timeline")
            timeline = helper.monthly_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

            # daily timeline
            st.title("Daily Timeline")
            daily_timeline = helper.daily_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

            # activity map
            st.title('Activity Map')
            col1, col2 = st.columns(2)

            with col1:
                st.header("Most busy day")
                busy_day = helper.week_activity_map(selected_user, df)
                fig, ax = plt.subplots()
                ax.bar(busy_day.index, busy_day.values, color='purple')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col2:
                st.header("Most busy month")
                busy_month = helper.month_activity_map(selected_user, df)
                fig, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values, color='orange')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            st.title("Weekly Activity Map")
            user_heatmap = helper.activity_heatmap(selected_user, df)
            fig, ax = plt.subplots()
            ax = sns.heatmap(user_heatmap)
            st.pyplot(fig)

            # group level
            if selected_user == "Overall":
                st.title("Most Busy User: ")
                x, y = helper.most_busy_user(df)
                fig, ax = plt.subplots()

                cola, colb = st.columns(2)

                with cola:
                    y.index += 1
                    st.dataframe(y)

                with colb:
                    ax.bar(x.index, x.values, color='Red')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)

            # word_cloud
            st.title("WordCloud")
            wordcloud_image = helper.create_word_cloud(selected_user, df)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud_image)
            st.pyplot(fig)

            # most_common_words
            st.title("Most Frequent Words: ")
            most_common_word = helper.most_common_words(selected_user, df)
            fig, ax = plt.subplots()
            ax.barh(most_common_word[0], most_common_word[1])
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

            # emoji analysis
            st.title("Most Common Emojis: ")
            emoji_df = helper.emoji_analysis(selected_user, df)
            if(not isinstance(emoji_df, int)):
                st.dataframe(emoji_df)
            else:
                st.title("No emoji detected!")

        if st.sidebar.button("Language & Sentiment Detection:"):
            eng_dataframe, noneng_dataframe, eng_count_message, non_eng_count_message = helper.message_language_count(selected_user,df)
            
            if(noneng_dataframe.shape[0]!=0):
                st.title("Messages other than english are shown below: ")
                if(noneng_dataframe.shape):
                    noneng_dataframe.index += 1
                    st.dataframe(noneng_dataframe)
        
            colx, coly = st.columns(2)

            with colx:
                st.header("Total number of english messages: ");
                st.title(eng_count_message)

                st.header("Total Number of non-english messages: ");
                st.title(non_eng_count_message)

            if selected_user == "Overall":
                st.title("User using non-english languages mostly: ")
                x, y = helper.most_busy_user(noneng_dataframe)
                y.index += 1
                fig, ax = plt.subplots()

                colm, coln = st.columns(2)

                with colm:
                    st.dataframe(y)

                with coln:
                    ax.bar(x.index, x.values, color='Red')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)

            st.title("Sentiment Analysis Results: ")
            st.title("Message dataset with corresponding sentiments: ")
            
            sentimentDataset = helper.message_sentiment_count(selected_user,df)
            st.dataframe(sentimentDataset)
            m, n = helper.seeSentiment(selected_user,sentimentDataset)
            fig, ax = plt.subplots()

            colx, coly = st.columns(2)

            with colx:
                    st.title("Messages parcentage based on sentiment type:")
                    st.dataframe(n)

            with coly:
                st.title("Graphical sentiment analysis: ")
                ax.bar(m.index, m.values, color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            joy_data,sadness_data,fear_data,anger_data,surprise_data,neutral_data,disgust_data,shame_data = helper.word_in_emotion(selected_user,sentimentDataset)
            

            

            if(joy_data.shape[0]!=0):
                st.title("WordCloud Joy")
                wordcloud_image_joy = helper.create_word_cloud(selected_user, joy_data)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud_image_joy)
                st.pyplot(fig)
            if(sadness_data.shape[0]!=0):
                st.title("WordCloud sadness")
                wordcloud_image_sadness = helper.create_word_cloud(selected_user, sadness_data)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud_image_sadness)
                st.pyplot(fig)
            if(fear_data.shape[0]!=0):
                st.title("WordCloud fear")
                wordcloud_image_fear = helper.create_word_cloud(selected_user, fear_data)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud_image_fear)
                st.pyplot(fig)
            if(anger_data.shape[0]!=0):
                st.title("WordCloud anger")
                wordcloud_image_anger = helper.create_word_cloud(selected_user, anger_data)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud_image_anger)
                st.pyplot(fig)
            if(surprise_data.shape[0]!=0):
                st.title("WordCloud surprise")
                wordcloud_image_surprise = helper.create_word_cloud(selected_user, surprise_data)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud_image_surprise)
                st.pyplot(fig)
            if(neutral_data.shape[0]!=0):
                st.title("WordCloud neutral")
                wordcloud_image_neutral = helper.create_word_cloud(selected_user, neutral_data)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud_image_neutral)
                st.pyplot(fig)
            if(disgust_data.shape[0]!=0):
                st.title("WordCloud disgust")
                wordcloud_image_disgust = helper.create_word_cloud(selected_user, disgust_data)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud_image_disgust)
                st.pyplot(fig)
            if(shame_data.shape[0]!=0):
                st.title("WordCloud shame")
                wordcloud_image_shame = helper.create_word_cloud(selected_user, shame_data)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud_image_shame)
                st.pyplot(fig)
    else:
        st.title("You have entered the wrong input!")

