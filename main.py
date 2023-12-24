import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import database as db
import pickle
from pathlib import Path
import streamlit_authenticator as stauth
import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, LabelBinarizer

def apply_preprocessing(data):
    # Initialize encoders
    label_encoder = LabelEncoder()
    label_binarizer = LabelBinarizer()

    # Encode 'job' column
    job_order = [['unknown','unemployed','student','retired','housemaid','services','blue-collar','technician','self-employed','management','admin.','entrepreneur']]
    ordinal_encoder_job = OrdinalEncoder(categories=job_order)
    data['job'] = ordinal_encoder_job.fit_transform(data[['job']])

    # Encode 'marital' column
    data['marital'] = label_encoder.fit_transform(data['marital'])

    # Encode 'education' column
    education_order = [['illiterate','unknown','basic.4y','basic.6y','basic.9y','high.school','university.degree','professional.course']]
    ordinal_encoder_education = OrdinalEncoder(categories=education_order)
    data['education'] = ordinal_encoder_education.fit_transform(data[['education']])

    # Encode 'default' column
    default_order = [['no','unknown','yes']]
    ordinal_encoder_default = OrdinalEncoder(categories=default_order)
    data['default'] = ordinal_encoder_default.fit_transform(data[['default']])

    # Encode 'housing' column
    housing_order = [['no','unknown','yes']]
    ordinal_encoder_housing = OrdinalEncoder(categories=housing_order)
    data['housing'] = ordinal_encoder_housing.fit_transform(data[['housing']])

    # Encode 'loan' column
    loan_order = [['no','unknown','yes']]
    ordinal_encoder_loan = OrdinalEncoder(categories=loan_order)
    data['loan'] = ordinal_encoder_loan.fit_transform(data[['loan']])

    # Encode 'contact' column using LabelBinarizer
    data['contact'] = label_binarizer.fit_transform(data['contact'])

    # Encode 'month' column
    data['month'] = label_encoder.fit_transform(data['month'])

    # Encode 'day_of_week' column
    data['day_of_week'] = label_encoder.fit_transform(data['day_of_week'])

    # Encode 'poutcome' column
    poutcome_order = [['nonexistent','failure','success']]
    ordinal_encoder_poutcome = OrdinalEncoder(categories=poutcome_order)
    data['poutcome'] = ordinal_encoder_poutcome.fit_transform(data[['poutcome']])
    data['duration_square']=data['duration'] ** 2
    data=data.drop(columns='nr.employed')

    return data

def make_predictions(input_csv):
    # Load the data with the correct delimiter
    data = pd.read_csv(input_csv, sep=',')
    data= data.drop(columns='key')    
    data = data.drop(columns=data.columns[0])
    data= data.drop(columns='time')
    # Load the pipeline (assuming the filename is always the same)
    pipeline_filename = 'xgb_pipeline_filename.joblib'
    loaded_pipeline = joblib.load(pipeline_filename)

    # Assuming you have new data for prediction
    new_data =data.iloc[[-1]]  # Use double brackets to select the first row as a DataFrame

    # Make predictions using the loaded pipeline
    predictions = loaded_pipeline.predict(new_data)

    # Return or use the predictions as needed
    return predictions

st.set_page_config(page_title="Future Bank",page_icon=":money_with_wings:",layout="wide")
#authentication
names=["Elif Arslan","admin"]
usernames=["elif.arslan","admin"]
file_path= Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords= pickle.load(file)
authenticator = stauth.Authenticate(names,usernames,hashed_passwords,"bank_page","abcdef", cookie_expiry_days=30)
name,authentication_status, username = authenticator.login("Login","main")
if authentication_status== False:
    st.error("username/password is incorrrect.")
if authentication_status== None:
        st.warning("Please enter your username and password.")
if authentication_status == True:
    base="dark"
    primaryColor="#2a2a4a"
    font="serif"
    #age,job,marital,education, default,balance,housing,loan,contact,day,month, duration,campaign,pdays, previous,poutcome,y 
    #change the background with html
    background_image = "https://www.cpomagazine.com/wp-content/uploads/2023/02/what-u-s-companies-can-learn-from-the-european-payment-scene_1500.jpg"
    background = f"""
        <style>
        .stApp {{
            background-image: url("{background_image}");
            background-size: cover;
            }}
        </style>
        """
    st.markdown(background, unsafe_allow_html=True)
    #add title
    st.title('Welcome to the FUTURE BANK '+":money_with_wings:")
    st.subheader('\"the bank of the future, the future of the banking\"')
    st.divider()
    authenticator.logout("logout","sidebar")
    st.sidebar.title(f"Welcome {name}")
    st.sidebar.markdown("Dear Team,")
    st.sidebar.markdown("As representatives of our bank,")
    st.sidebar.markdown("it's crucial to maintain a high standard of professionalism and courtesy when reaching out to our customers." )
    st.sidebar.markdown("Here are some guidelines to ensure polite and respectful communication during customer calls:")
    st.sidebar.markdown("Greeting: Always begin the call with a warm and friendly greeting.")
    st.sidebar.markdown("A simple -Hello, this is [Name] from the Future Bank. How may I assist you today?- sets a positive tone.")
    st.sidebar.markdown("Active Listening: Pay close attention to the customers concerns and needs.") 
    st.sidebar.markdown("Show empathy by actively listening and acknowledging their inquiries.")
    st.sidebar.markdown("Polite Language: Use polite language throughout the conversation.") 
    st.sidebar.markdown("Please and thank you go a long way in making customers feel valued.")
    st.sidebar.markdown("Professional Tone: Maintain a professional tone and refrain from using slang or informal language.")
    st.sidebar.markdown("Solution-Oriented Approach: Focus on finding solutions to the customers concerns. ")
    st.sidebar.markdown("Offer assistance and options to resolve issues effectively.Closing: End the call on a positive note,expressing gratitude for their time and offering further assistance if needed.") 
    st.sidebar.markdown("Remember, every interaction with a customer reflects our commitment to exceptional service.")
    st.sidebar.markdown(" Your dedication to polite and respectful communication is greatly appreciated.Thank you for your attention to these guidelines.")
    st.sidebar.markdown("Best regards,")
    st.sidebar.markdown("Elif Arslan")
    st.sidebar.markdown("Bank Manager")

    selected = option_menu(
    menu_title=None,
    options=["enter client data","data view",],
    icons=["pencil-fill", "bar-chart-fill"],  
    orientation="horizontal",
    )
    #option 1:
    if selected=="enter client data":
        with st.form("form"):
            st.divider()
            age = st.number_input("Enter client's age:", min_value=0, max_value=150, step=1)
            st.divider()
            occupations = [
                'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
                'retired', 'self-employed', 'services', 'student', 'technician',
                'unemployed', 'unknown'
            ]    
            job = st.selectbox('Select job:', occupations)
            st.divider()
            marital_status = ['divorced','married','single','unknown']
            marital = st.selectbox('Select martial status:',marital_status)
            st.divider()
            education_level = ["basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown"]
            education=st.selectbox('Select education level:',education_level)
            st.divider()
            default = st.radio("Has credit in default?", ('yes', 'no','unknown'))
            st.divider()
            housing= st.radio("Has housing loan?", ('yes', 'no','unknown'))
            st.divider()
            loan= st.radio("Has personal loan?", ('yes', 'no','unknown'))
            st.divider()
            contact_type=['cellular','telephone']
            contact = st.selectbox('Contact communication type:',contact_type)
            st.divider()
            months = [
                'jan', 'feb', 'mar', 'apr', 'may', 'jun',
                'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
            ]
            month = st.selectbox('Last contact month of year:',months)
            st.divider()
            days=["mon","tue","wed","thu","fri"]
            day_of_week=st.selectbox('Last contact day:', days)
            st.divider()
            duration = st.number_input("Enter the last contact duration in seconds:", min_value=0)
            st.divider()
            campaign = st.number_input('Number of contacts performed during this campaign and for this client:', min_value = 0)
            st.divider()
            pdays = st.number_input('Number of days that passed by after the client was last contacted from a previous campaign:',step=1)
            st.divider()
            previous = st.number_input('Number of contacts performed before this campaign and for this client:',min_value = 0)
            st.divider()
            poutcomes= ['failure','nonexistent','success']
            poutcome = st.selectbox('Outcome of the previous marketing campaign:',poutcomes)
            st.divider()
            emp_var_rate= st.number_input('Employment variation rate - quarterly indicator ')
            st.divider()
            cons_price_idx = st.number_input('Consumer price index - monthly indicator')
            st.divider()
            cons_conf_idx= st.number_input('Consumer confidence index - monthly indicator:')
            st.divider()
            euribor3m = st.number_input('Euribor 3 month rate - daily indicator:')
            st.divider()
            nr_employed = st.number_input('Number of employees - quarterly indicator:')
            st.divider()            
            submitted = st.form_submit_button("add client data")
        if submitted:
            #insert to database
            db.insert_data(age,job,marital,education, default,housing,loan,contact,month,day_of_week, duration,campaign,pdays, previous,poutcome,emp_var_rate,cons_price_idx,cons_conf_idx,euribor3m,nr_employed)
            st.write("client data is added.")
            show=db.fetch_all_data() 
            df = pd.DataFrame(show)
            df = df.sort_values(by='time')
            file_path = 'data.csv'
            df.to_csv(file_path, index=True)
            result=make_predictions('data.csv')            
            st.title("Has the client subscribed a term deposit?")
            if result==0:
                st.subheader("no")
            if result==1:
                st.subheader("yes")
            
            

                       
#option 2:
    if selected=="data view" :
        show=db.fetch_all_data() 
        df = pd.DataFrame(show)
        st.dataframe(df)

    
