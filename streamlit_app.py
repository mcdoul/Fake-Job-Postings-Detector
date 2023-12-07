import streamlit as st
import joblib
import pandas as pd
import re
from streamlit_option_menu import option_menu
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


base_pipeline = joblib.load('base_pipeline.joblib')
meta_pipeline = joblib.load('meta_pipeline_with_location.joblib')
TEXT_FEATURES = ['title', 'company_profile', 'description', 'requirements', 'benefits']


industries = ['Not Provided',
 'Accounting',
 'Airlines/Aviation',
 'Alternative Dispute Resolution',
 'Animation',
 'Apparel & Fashion',
 'Architecture & Planning',
 'Automotive',
 'Aviation & Aerospace',
 'Banking',
 'Biotechnology',
 'Broadcast Media',
 'Building Materials',
 'Business Supplies and Equipment',
 'Capital Markets',
 'Chemicals',
 'Civic & Social Organization',
 'Civil Engineering',
 'Commercial Real Estate',
 'Computer & Network Security',
 'Computer Games',
 'Computer Hardware',
 'Computer Networking',
 'Computer Software',
 'Construction',
 'Consumer Electronics',
 'Consumer Goods',
 'Consumer Services',
 'Cosmetics',
 'Defense & Space',
 'Design',
 'E-Learning',
 'Education Management',
 'Electrical/Electronic Manufacturing',
 'Entertainment',
 'Environmental Services',
 'Events Services',
 'Executive Office',
 'Facilities Services',
 'Farming',
 'Financial Services',
 'Fishery',
 'Food & Beverages',
 'Food Production',
 'Fund-Raising',
 'Furniture',
 'Gambling & Casinos',
 'Government Administration',
 'Government Relations',
 'Graphic Design',
 'Health, Wellness and Fitness',
 'Higher Education',
 'Hospital & Health Care',
 'Hospitality',
 'Human Resources',
 'Import and Export',
 'Individual & Family Services',
 'Industrial Automation',
 'Information Services',
 'Information Technology and Services',
 'Insurance',
 'International Trade and Development',
 'Internet',
 'Investment Banking',
 'Investment Management',
 'Law Enforcement',
 'Law Practice',
 'Legal Services',
 'Leisure, Travel & Tourism',
 'Libraries',
 'Logistics and Supply Chain',
 'Luxury Goods & Jewelry',
 'Machinery',
 'Management Consulting',
 'Maritime',
 'Market Research',
 'Marketing and Advertising',
 'Mechanical or Industrial Engineering',
 'Media Production',
 'Medical Devices',
 'Medical Practice',
 'Mental Health Care',
 'Military',
 'Mining & Metals',
 'Motion Pictures and Film',
 'Museums and Institutions',
 'Music',
 'Nanotechnology',
 'Nonprofit Organization Management',
 'Oil & Energy',
 'Online Media',
 'Outsourcing/Offshoring',
 'Package/Freight Delivery',
 'Packaging and Containers',
 'Performing Arts',
 'Pharmaceuticals',
 'Philanthropy',
 'Photography',
 'Plastics',
 'Primary/Secondary Education',
 'Printing',
 'Professional Training & Coaching',
 'Program Development',
 'Public Policy',
 'Public Relations and Communications',
 'Public Safety',
 'Publishing',
 'Ranching',
 'Real Estate',
 'Religious Institutions',
 'Renewables & Environment',
 'Research',
 'Restaurants',
 'Retail',
 'Security and Investigations',
 'Semiconductors',
 'Shipbuilding',
 'Sporting Goods',
 'Sports',
 'Staffing and Recruiting',
 'Telecommunications',
 'Textiles',
 'Translation and Localization',
 'Transportation/Trucking/Railroad',
 'Utilities',
 'Venture Capital & Private Equity',
 'Veterinary',
 'Warehousing',
 'Wholesale',
 'Wine and Spirits',
 'Wireless',
 'Writing and Editing']
functions = ['Not Provided',
 'Accounting/Auditing',
 'Administrative',
 'Advertising',
 'Art/Creative',
 'Business Analyst',
 'Business Development',
 'Consulting',
 'Customer Service',
 'Data Analyst',
 'Design',
 'Distribution',
 'Education',
 'Engineering',
 'Finance',
 'Financial Analyst',
 'General Business',
 'Health Care Provider',
 'Human Resources',
 'Information Technology',
 'Legal',
 'Management',
 'Manufacturing',
 'Marketing',
 'Product Management',
 'Production',
 'Project Management',
 'Public Relations',
 'Purchasing',
 'Quality Assurance',
 'Research',
 'Sales',
 'Science',
 'Strategy/Planning',
 'Supply Chain',
 'Training',
 'Writing/Editing',
 'Other'
 ]
locations = [
    'Not Provided', 'AE', 'AL', 'AM', 'AR', 'AT', 'AU', 'BD', 'BE', 'BG', 'BH', 
    'BR', 'BY', 'CA', 'CH', 'CL', 'CM', 'CN', 'CO', 'CY', 'CZ', 'DE', 'DK', 'EE', 
    'EG', 'ES', 'FI', 'FR', 'GB', 'GH', 'GR', 'HK', 'HR', 'HU', 'ID', 'IE', 'IL', 
    'IN', 'IQ', 'IS', 'IT', 'JM', 'JP', 'KE', 'KH', 'KR', 'KW', 'KZ', 'LK', 'LT', 
    'LU', 'LV', 'MA', 'MT', 'MU', 'MX', 'MY', 'NG', 'NI', 'NL', 'NO', 'NZ', 'PA', 
    'PE', 'PH', 'PK', 'PL', 'PT', 'QA', 'RO', 'RS', 'RU', 'SA', 'SD', 'SE', 'SG', 
    'SI', 'SK', 'SV', 'TH', 'TN', 'TR', 'TT', 'TW', 'UA', 'UG', 'US', 'VI', 'VN', 
    'ZA', 'ZM'
]


def process_text(post):
    def clean_text(series):
        def remove_URL(text):
            url_pattern1 = re.compile(r'https?://\S+')  # remove any url starting with "http://" or "https://"
            url_pattern2 = re.compile(r'#URL_\S+#')
            text = url_pattern1.sub(r'', text)
            text = url_pattern2.sub(r'', text)
            return text

        def remove_email(text):
            email_pattern = re.compile(r'#EMAIL_\S+#')
            text = email_pattern.sub(r'', text)
            return text

        def remove_phone(text):
            phone_pattern = re.compile(r'#PHONE_\S+#')
            text = phone_pattern.sub(r'', text)
            return text

        def remove_punctuation(text):
            table = str.maketrans(',./;', '    ');
            return text.translate(table)

        def decontracted(phrase):
            """Reference: https://www.kaggle.com/code/gordotron85/nlp-text-classification-linear-models-vs-bert?scriptVersionId=39182957&cellId=38"""
            # specific
            phrase = re.sub(r"won't", "will not", phrase)
            phrase = re.sub(r"can't", "can not", phrase)

            # general
            phrase = re.sub(r"n't", " not", phrase)
            phrase = re.sub(r"'re", " are", phrase)
            phrase = re.sub(r"'s", " is", phrase)
            phrase = re.sub(r"'d", " would", phrase)
            phrase = re.sub(r"'ll", " will", phrase)
            phrase = re.sub(r"'t", " not", phrase)
            phrase = re.sub(r"'ve", " have", phrase)
            phrase = re.sub(r"'m", " am", phrase)
            return phrase
        
        def final_preprocess(text):
            text = text.replace('\\r', ' ')
            text = text.replace('\\n', ' ')
            text = re.sub('[^A-Za-z0-9]+', ' ', text)
            text = ' '.join(e for e in text.split() if e.lower() not in stopwords.words("English"))
            text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add a space before every uppercase letter that follows a lowercase
            # print('Added space before uppercase: ', text)
            text = text.lower()
            return text

        series = series.apply(remove_URL)
        series = series.apply(remove_email)
        series = series.apply(remove_phone)
        series = series.apply(remove_punctuation)
        series = series.apply(decontracted)
        series = series.apply(final_preprocess)
        return series
    
    df = post
    df_temp = pd.DataFrame()
    df_temp['raw_text'] = df[['title', 'company_profile', 'description', 'requirements', 'benefits']].apply(lambda x: ' '.join(x), axis = 1)
    df['Num_of_URLs'] = df_temp['raw_text'].str.count('#URL_') 
    df['Num_of_Phones'] = df_temp['raw_text'].str.count('#PHONE_') 
    df['Num_of_Emails'] = df_temp['raw_text'].str.count('#EMAIL_')
    
    clean_text_features = []
    for feature in TEXT_FEATURES:
        df_temp['clean_' + feature] = clean_text(df[feature])
        clean_text_features.append('clean_' + feature) 
        
    df['clean_joint_text'] = df_temp[clean_text_features].apply(lambda x: ' '.join(x), axis = 1)
    df['text_len'] = df['clean_joint_text'].apply(lambda text: len(text))

    return df


#Layout
st.set_page_config(
    page_title="",
    layout="wide",
    initial_sidebar_state="expanded")

#Data Pull and Functions
st.markdown("""
<style>
.big-font {
    font-size:80px !important;
}
</style>
""", unsafe_allow_html=True)



#Options Menu
with st.sidebar:
    selected = option_menu('MENU', ["Home", 'Fake Job Posts Prediction','Fake Job Posts Analysis'], 
        icons=['house','search','bi-graph-up'],menu_icon='list', default_index=0)

#Home Page
if selected=="Home":
    #Header
    st.title('Welcome to Fake Job Postings Detector!')
    st.subheader('*A efficient tool for job seekers to distinguish fake job postings.*')

    st.divider()

    #Introduction
    st.header('Introduction')
    st.markdown(
        """
        These days, many job seekers encounter fraudulent job postings. Some companies post these listings not 
        to hire, but to collect personal data, and some are outright scams for application fee. 
        CBS News even did a piece on this, showing how common the issue is.

        That’s where our project, the Fake Job Postings Detector, comes in. By training a binary classification model, 
        our Fake Job Postings Detector can address the critical challenge of distinguishing the real job offers from the fake ones. 
        Aiming to make job searches safer and more reliable, our tool is designed to help job seekers not only save time but also 
        protect their personal information and money effectively.
        """
        )   
    st.info("Learn more about this topic in a detailed article from CBS News: [Read Here](https://www.cbsnews.com/news/job-openings-fake-listings-ads-federal-reserve-jolts/)")

    st.divider()


    # Use Case
    st.header('Use Cases')
    st.markdown(
        """
        - _Thinking About a Career Move?_
        - _Ready to Land that Dream Job?_
        - _Graduating and Looking for Your First Job?_
        - _Just Curious About What's Out There?_
        """
        )           

    st.divider()

    #Team members
    st.header('Team Members')
    st.write('**Chuan-Hsin Chen:**    Data preprocessing, Model training')
    st.write('**Hongchao Fang:**    Model training')
    st.write('**Jin Zhang:**    Data analysis')
    st.write('**Yen-Yu Yang:**    Web app development')

    st.divider()

    # References
    st.header('References')
    col1,col2,col3=st.columns(3)
    col1.subheader('Source')
    col2.subheader('Description')
    col3.subheader('Link')
    with st.container():
        col1,col2,col3=st.columns(3)
        col1.write('Real / Fake Job Posting Prediction Data Set')
        col2.write('*Real and fake job postings data set*')
        col3.write('https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction')
    
    with st.container():
        col1,col2,col3=st.columns(3)
        col1.write('Fake job listings are a growing problem in the labor market')
        col2.write('*CBS News for fake job posting*')
        col3.write('https://www.cbsnews.com/news/job-openings-fake-listings-ads-federal-reserve-jolts/')
    
    with st.container():
        col1,col2,col3=st.columns(3)
        col1.write('SimiLo')
        col2.write('*Streamlit app for layout referencing*')
        col3.write('https://similobeta2.streamlit.app/#welcome-to-similo')
    

    
#Prediction Page
if selected=="Fake Job Posts Prediction":

    st.header('Please enter the job information')

    with st.form("job_info_form"):
        job_title = st.text_input("Job Title")
        location = st.selectbox("Location", locations)

        with st.container():
            col1,col2=st.columns(2)
            department = col1.selectbox("Department", ["Provided", "Not Provided"])
            salary_range = col2.selectbox("Salary Range", ["Provided", "Not Provided"])

        with st.container():
            col1,col2=st.columns(2)
            company_profile = col1.text_area("Company Profile")  
            description = col2.text_area("Job Description")

        with st.container():
            col1,col2=st.columns(2)
            requirements = col1.text_area("Requirements")
            benefits = col2.text_area("Benefits")

        with st.container():
            col1,col2, col3=st.columns(3)
            has_company_logo = col1.checkbox("Has Company Logo")
            has_questions = col2.checkbox("Has Questions")
            telecommuting = col3.checkbox("Telecommuting")

        
        employment_type = col1.selectbox("Employment Type", ['Not Provided', 'Other', 'Full-time', 'Part-time', 'Contract', 'Temporary'])
        
        with st.container():
            col1,col2=st.columns(2)
            required_experience = col1.selectbox("Required Experience", ['Not Provided', 'Not Applicable', 'Internship', 'Mid-Senior level', 'Associate', 'Entry level', 'Executive', 'Director'])
            required_education =col2.selectbox("Required Education", ['Unspecified','High School or equivalent','Associate Degree',"Bachelor's Degree","Master's Degree",'Other'])
        with st.container():
            col1,col2=st.columns(2)
            industry = col1.selectbox("Industry", industries)
            function = col2.selectbox("Function", functions)
            
        submitted = st.form_submit_button("Submit")

        if submitted:
            text_features = {
                'title': "oops" if job_title == "" else job_title,
                'company_profile': "oops" if company_profile == "" else company_profile,
                'description': "oops" if description == "" else description,
                'requirements': "oops" if requirements == "" else requirements,
                'benefits': "oops" if benefits == "" else benefits,
            }

            # Predict with base model
            text_df = pd.DataFrame(text_features, index=[0])
            processed_text = process_text(text_df)
            stage1_pred = base_pipeline.predict(processed_text)[0]

            non_text_features = {
                'location' : location,
                'department': 0 if department == "Not Provided" else 1,
                'salary_range': 0 if salary_range == "Not Provided" else 1,
                'telecommuting': int(telecommuting),
                'has_company_logo': int(has_company_logo),
                'has_questions': int(has_questions),
                'employment_type': employment_type,
                'industry': industry,
                'required_experience': required_experience,
                'required_education': required_education,
                'function' : function,
                'stage1_pred' : stage1_pred,
                'Num_of_URLs': processed_text['Num_of_URLs'].iloc[0],
                'Num_of_Phones': processed_text['Num_of_Phones'].iloc[0],
                'Num_of_Emails': processed_text['Num_of_Emails'].iloc[0],
                'text_len': processed_text['text_len'].iloc[0]
            }

            # Predict with meta model
            non_text_df = pd.DataFrame([non_text_features])
            final_prediction = meta_pipeline.predict(non_text_df)

            # Display the result
            # st.write("Prediction:", "Fraudulent" if final_prediction[0] == 1 else "Not Fraudulent")
            if final_prediction[0] == 1:
                st.error("This is a **Fake Job Post**")
                st.markdown('**Suggestion:** Be cautious! This job posting has characteristics of a fraudulent posting. Verify the company details and proceed with care.')
            else:
                st.success("This is a **Legit Job Post**")
                st.markdown('**Suggestion:** This job posting appears to be legitimate. However, always ensure to do your due diligence before applying.')

#Analysis Page - 
if selected=='Fake Job Posts Analysis':
    st.header('Fake Job Posts Analysis')
    
    
    # read data
    df = pd.read_csv('fake_job_postings.csv', index_col = False)
    df['employment_type'] = df['employment_type'].fillna("Employment Unavailable")
    df['required_experience'] = df['required_experience'].fillna("Experience Unavailable")
    df['required_education'] = df['required_education'].fillna("Unspecified")
    # text_features = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    # df[text_features] = df[text_features].fillna('')
    # processed_text = process_text(df)
    dark_blue, light_blue, bg, yellow, orange, pink, red, green, light_green = "#2B65C1", "#94C8FA","#E7F4FE", "#F8D47D", "#EF8D34", "#F2ABA9", "#EA3A2D", "#54AB9A", "#96EBA4"


    # plot1 - Fake Job Ratio & Total Number of Jobs of Top 10 States with the Most Job Postings
    st.write("**Fake Job Ratio & Total Number of Jobs of Top 10 States with the Most Job Postings**")
    loc_split = []
    for loc in df.location:
        loc = str(loc)
        loc_split.append(loc.split(','))

    loc_split = pd.DataFrame(loc_split)
    loc_split = loc_split[[1, 2]]
    loc_split = loc_split.rename(columns={1: "state", 2: "city"})
    df = df.join(loc_split)
    df['state_city'] = df['state'] + ", " + df['city']
    df.city = df.city.str.strip()
    df.state = df.state.str.strip()

    top_10_states = set(df['state'].value_counts().iloc[1:11].index.tolist())
    df_temp = df[['state', 'fraudulent']].loc[df['state'].isin(top_10_states)]
    df_temp = df_temp.groupby('state')['fraudulent'].value_counts().unstack()
    df_temp.reset_index(inplace=True)
    df_temp.fillna(0, inplace=True)

    df_temp['fake_job_ratio'] = df_temp[1] / (df_temp[0] + df_temp[1]) * 100
    df_temp['total_job_cnt'] = df_temp[0] + df_temp[1]

    fig, ax1 = plt.subplots(figsize=(16,6))
    width = 0.2
    ind = np.arange(len(df_temp))  

    ax1.bar(ind - width/2, df_temp['fake_job_ratio'], width, color='orange', label='Fake Job Ratio')
    ax2 = ax1.twinx()

    ax2.bar(ind + width/2, df_temp['total_job_cnt'], width, color=light_blue, label='Total Job Count')

    ax1.set_xlabel('State', fontsize=20)
    ax1.set_ylabel('Fake Job Percentage(%)', fontsize=20)
    ax2.set_ylabel('Number of Jobs', fontsize=20)

    ax1.set_xticks(ind)
    ax1.set_xticklabels(df_temp['state'], rotation=45, fontsize=15)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper left')

    st.pyplot(fig)


    #plot2 - Fake Job Ratio & Total Number of Jobs of Top 10 Cities with the Most Job Postings
    st.write("**Fake Job Ratio & Total Number of Jobs of Top 10 Cities with the Most Job Postings**")
    top_10_cities = set(df['city'].value_counts().iloc[1:11].index.tolist())
    df_temp = df[['city', 'fraudulent']].loc[df['city'].isin(top_10_cities)]

    df_temp = df_temp.groupby('city')['fraudulent'].value_counts().unstack()
    df_temp.reset_index(inplace=True)
    df_temp.fillna(0, inplace=True)

    df_temp['fake_job_ratio'] = df_temp[1] / (df_temp[0] + df_temp[1]) * 100
    df_temp['total_job_cnt'] = df_temp[0] + df_temp[1]

    fig, ax1 = plt.subplots(figsize=(16,6))
    width = 0.2

    ax1.bar(df_temp['city'], df_temp['fake_job_ratio'], color=orange, width=-width, align='edge', label='Fake Job Ratio')
    ax2 = ax1.twinx()
    ax2.bar(df_temp['city'], df_temp['total_job_cnt'], color=green, width=width, align='edge', label='Total Job Count')

    ax1.set_xlabel('City', fontsize=20)
    ax1.set_xticks(range(len(df_temp['city'])))
    ax1.set_xticklabels(df_temp['city'], rotation=45, fontsize=15)

    ax1.set_ylabel('Fake Job Ratio(%)', fontsize=20)
    ax2.set_ylabel('Number of Jobs', fontsize=20)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    ax1.legend(h1+h2, l1+l2, loc='upper left')

    st.pyplot(fig)


    # plot3 - Number of Job Postings based on Locations
    st.write("**Number of Job Postings based on Locations**")
    def split_country(location):
        if isinstance(location, str):
            return location.split(',')[0].strip()
        else:
            return 'Unknown'

    df['Country'] = df['location'].apply(split_country)
    Country = dict(df[df['Country'] != 'Unknown']['Country'].value_counts()[:20])

    fig, ax = plt.subplots(figsize=(8,6))
    ax.bar(Country.keys(), Country.values(), color=[yellow])
    ax.set_xlabel("Country", fontsize=12)
    ax.set_ylabel("Number of Jobs", fontsize=12)

    st.pyplot(fig)
    

    # plot4 - Number of Non Fraud and Fraud Job Postings based on Education Requirement
    st.write("**Number of Non Fraud and Fraud Job Postings based on Education Requirement**")
    
    fig, ax = plt.subplots()
    colors = [light_green, orange]
    sns.countplot(ax=ax, x='required_education', data=df, hue='fraudulent', palette=colors)
    ax.set_xticklabels([
        "Bachelor's Degree", "Master's Degree", 'High School or equivalent', 'Unspecified',
        'Some College Coursework Completed', 'Vocational', 'Certification', 'Associate Degree',
        'Professional', 'Doctorate', 'Some High School Coursework', 'Vocational - Degree',
        'Vocational - HS Diploma'], rotation=90)
    legend_labels = ['Not Fraudulent', 'Fraudulent']
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, legend_labels, loc='upper right')

    st.pyplot(fig)
    

    # plot5 - Number of Non Fraud and Fraud Job Postings based on Employment Type
    st.write("**Number ofNon Fraud and Fraud Job Postings based on Employment Type**")
    colors = [green, orange]
    def count_plot(df, col, hue):
        fig, ax = plt.subplots(figsize=(15,5))
        sns.countplot(x=col, data=df, hue=hue, ax=ax,palette=colors)
        legend_labels = ['Not Fraudulent', 'Fraudulent']
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, legend_labels, title=hue, loc='upper right')
        st.pyplot(fig)

    count_plot(df, 'employment_type', 'fraudulent')


    #plot 6 - Number of Fraud Job Postings based on Required Experience
    st.write("**Number of Fraud Job Postings based on Required Experience**")
    df_fraud = df[df['fraudulent'] == 1]
    fig_1 = px.bar(
        df_fraud,
        x="required_experience",
        y="fraudulent",
        color="required_experience",
        pattern_shape="required_experience",
        pattern_shape_sequence=[".", "x", "+", "-", "x", '|']
    )
    st.plotly_chart(fig_1)
    
    
    # plot 7 - Top 10 Industries Represented in this Dataset
    st.write("**Top 10 Industries Represented in this Dataset**")
    industry = df.industry.value_counts()[0:10]
    colors=[light_blue]

    fig, ax = plt.subplots(figsize=(12, 6))
    industry.plot(kind='barh', ax=ax, color = colors)
    ax.set_xlabel('Count')

    st.pyplot(fig)


    # plot 8 - Number of Fraud Job Postings based on industry
    st.write("**Number of Fraud Job Postings based on industry**")
    df_fraud[['job_id', 'industry']].groupby('industry').count().sort_values('job_id')
    fig_4 = px.bar(
        df_fraud,
        y="industry",
        x="fraudulent",
        color="industry",
        pattern_shape="industry",
        pattern_shape_sequence=[".", "x", "+", "-", "x", '|']
    )
    st.plotly_chart(fig_4)


    # plot 9 - Top 10 Business Functions Represented in this Dataset
    st.write("**Top 10 Business Functions Represented in this Dataset**")
    function = df.function.value_counts()[0:10]
    colors=[pink]

    fig, ax = plt.subplots(figsize=(12, 6))
    function.plot(kind='barh', ax=ax, color=colors)
    ax.set_xlabel('Count')

    st.pyplot(fig)
    
    
    # plot10 - Number of Fraud Job Postings based on function
    st.write("**Number of Fraud Job Postings based on function**")
    fig_5 = px.bar(
        df_fraud,
        y="function",
        x="fraudulent",
        color="function",
        pattern_shape="function",
        pattern_shape_sequence=[".", "x", "+", "-", "x", '|']
    )
    st.plotly_chart(fig_5)

    
    # plot11 - Distribution of Text Length
    st.write("**Distribution of Text Length**")
    df['text'] = df['title'].fillna('') + ' ' + df['location'].fillna('') + ' ' + df['department'].fillna('') + ' ' + df['company_profile'].fillna('') + ' ' + df['description'].fillna('') + ' ' + df['requirements'].fillna('') + ' ' + df['benefits'].fillna('') + ' ' + df['employment_type'].fillna('') + ' ' + df['required_education'].fillna('') + ' ' + df['industry'].fillna('') + ' ' + df['function'].fillna('')
    
    def length(x):
        if pd.isna(x):
            return 0
        return len(str(x))

    df['length_text'] = df['text'].apply(length)

    real_job_text_len = df.loc[df['fraudulent'] == 0]['length_text']
    fake_job_text_len = df.loc[df['fraudulent'] == 1]['length_text']

    fig, ax = plt.subplots()
    real_job_text_len.plot(kind='density', color=light_blue, lw=3, ax=ax, label='Non-Fraudulent')
    fake_job_text_len.plot(kind='density', color=dark_blue, lw=3, ax=ax, label='Fraudulent')
    ax.set_xlim(0, 10000)
    ax.legend()

    st.pyplot(fig)
    

    #plot12 - Boolean Feature's Value Distribution
    st.write("**Boolean Feature's Value Distribution**")
    def process_boolean_feature(feature, yes_label, no_label, ax):
        gb = df[[feature, 'fraudulent']].groupby('fraudulent').value_counts(normalize=True)
        df_temp = pd.DataFrame()
        df_temp['yes_ratio'] = pd.Series([gb.loc[(0,1)], gb.loc[(1,1)]])
        df_temp['no_ratio'] = pd.Series([gb.loc[(0,0)], gb.loc[(1,0)]])

        width = 0.2
        bottom = np.zeros(2)
        p1 = ax.bar(x=['Non-Fraud', 'Fraud'], height=df_temp['yes_ratio'], bottom=bottom, color=yellow, label=yes_label)
        bottom += df_temp['yes_ratio']
        p2 = ax.bar(x=['Non-Fraud', 'Fraud'], height=df_temp['no_ratio'], bottom=bottom, color="#D9D9D9", label=no_label)

        ax.bar_label(p1, fmt="{:.0%}", label_type='center', fontsize=15)
        ax.bar_label(p2, fmt="{:.0%}", label_type='center', fontsize=15)

        ax.set_xticklabels(labels=['Non-Fraudulent', 'Fraudulent'])
        ax.set_ylabel('Job Posts Ratio')

        ax.set_title(feature + " w.r.t Fraudulent or Not")
        ax.legend()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    ax1, ax2, ax3 = axes

    process_boolean_feature('has_questions', 'Has questions', 'No questions', ax1)
    process_boolean_feature('has_company_logo', 'Has company logo', 'No company logo', ax2)
    process_boolean_feature('telecommuting', 'Yes', 'No', ax3)

    st.pyplot(fig)


    # plot 13 - Variation of URLs, Phones, Emails with Fraudulent
    df['text_data'] = df['company_profile'].fillna('') + ' ' + df['description'].fillna('') + ' ' + df['requirements'].fillna('') + ' ' + df['benefits'].fillna('')
    df['Number_of_URLs'] = df['text_data'].str.count('#URL_')
    df['Number_of_Phones'] = df['text_data'].str.count('#PHONE_')
    df['Number_of_Emails'] = df['text_data'].str.count('#EMAIL_')
    # df['Number_of_URLs'] = processed_text['Num_of_URLs'].iloc[0]
    # df['Number_of_Phones'] = processed_text['Num_of_Phones'].iloc[0]
    # df['Number_of_Emails'] = df['textual_data'].str.count('#EMAIL_')
    # st.header('3D Scatter Plot: Variation of URLs, Phones, Emails with Fraudulent')
    fig = px.scatter_3d(df,
                        x="Number_of_URLs",
                        y="Number_of_Phones",
                        z="Number_of_Emails",
                        color="fraudulent",
                        size_max=10,
                        title="Variation of URLs, Phones, Emails with Fraudulent")
    st.plotly_chart(fig, use_container_width=True)


    # plot 14 -17
    image1 = Image.open('non fraud company.png')
    image2 = Image.open('fraud company.png')
    image3 = Image.open('non fraud benefits.png')
    image4 = Image.open('fraud benefits.png')

    st.write("**Company Profile Words for Non-Fraud Jobs**")
    st.image(image1)

    st.write("**Company Profile Words for Fraud Jobsn**")
    st.image(image2)

    st.write("**Benefits Words for Non-Fraud Jobs**")
    st.image(image3)

    st.write("**Benefits Words for Fraud Jobs**")
    st.image(image4)


