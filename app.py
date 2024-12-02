# from tkinter import Image
from PIL import Image
# pip install nltk
import streamlit as st
import nltk
# import streamlit as st
import pandas as pd
import re
# import nltk
import numpy as np
import matplotlib.pyplot as plt
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import fitz  # PyMuPDF for PDF handling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load the Dataset
df = pd.read_csv('cleaned_file.csv')


relevant_keywords = ['iot', 'cybersecurity', 'machine learning', 'ai', 'data science', 'blockchain', 'developer', 'engineer', 'software', 'embedded', 'technologist']
# Preprocessing and Cleaning Functions
def clean_text(txt):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    clean_text = re.sub(r'http\S+\s|RT|cc|#\S+\s|@\S+|[^\x00-\x7f]', ' ', txt)
    clean_text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip().lower()
    tokens = word_tokenize(clean_text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalpha()]
    return ' '.join(tokens)



df['cleaned_job_title'] = df['Job Title'].apply(clean_text)
df['cleaned_skills'] = df['Skills'].apply(clean_text)

# Remove duplicate rows based on cleaned job titles and skills
df = df.drop_duplicates(subset=['cleaned_job_title', 'cleaned_skills'])

# Filter out irrelevant job titles based on keywords (enhanced filtering)
def is_relevant_job(title):
    return any(keyword in title for keyword in relevant_keywords) and ('marine' not in title)

# Keep only relevant job titles
df['relevant'] = df['cleaned_job_title'].apply(is_relevant_job)
df = df[df['relevant']]  # Filter the dataframe to keep only relevant rows


# Extract text from PDF using PyMuPDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        st.error(f"Failed to extract text from the PDF: {e}")
    return text

# Initialize TF-IDF Vectorizer and KNN Model
vectorizer = TfidfVectorizer(max_features=5000)
job_descriptions = df['Skills'].apply(clean_text).tolist()
X = vectorizer.fit_transform(job_descriptions)

knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(X)

st.markdown("""<style>
    body {
        background-color: grey;  /* Light, clean background */
        color: pink;
        font-family: Georgia, 'Times New Roman', Times, serif;

    }

    .title {
        text-align: center;
        color: blue;  /* Bright Blue */
        font-size: 30px;
        font-weight: 700;
        text-transform: uppercase;
        font-family: Georgia, 'Times New Roman', Times, serif;


    }

    .subtitle {
        text-align: left;
        font-size: 20px;
        color: red;
        font-weight: 650;
        font-family: Arial, Helvetica, sans-serif;

    }

    .footer {
        text-align: center;
        padding: 20px;
        color: white;
        background-color: #1d61b4;
        font-size: 14px;
    }
    .job-list {
        display: grid;
        grid-template-columns: repeat(3, 1fr);  /* 3 items per row */
        gap: 20px;
        margin-top: 20px;
    }

    /* Job Item Style */
    .job-item {
        background-color: #ff6f61;  /* Bright Coral */
        color: white;
        border-radius: 15px;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
        box-sizing: border-box;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        cursor: pointer;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        padding: 20px;
    }

    .job-item:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        background-color: #ff4b39;  /* Darker Coral */
    }

    /* Load More Button */
    .load-more-btn {
        background-color: #64b5f6;  /* Light Blue */
        color: white;
        font-size: 20px;
        font-weight: bold;
        border-radius: 50px;
        text-align: center;
        cursor: pointer;
        border: none;
        display: block;
        margin-left: auto;
        margin-right: auto;
        animation: pulse 1s infinite;
        padding: 15px 30px;
    }

    .load-more-btn:hover {
        background-color: #039be5;  /* Deep Blue */
        transform: scale(1.05);
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }

    /* Table Styling */
    table {
        width: 100%;
        text-align: center;
        border-collapse: collapse;
        margin-top: 20px;
    }

    table, th, td {
        border: 1px solid #5e92f3;  /* Soft Blue */
        border-radius: 5px;
    }

    th, td {
        padding: 12px;
        background-color: #f3f9fc;  /* Light Blue Background */
    }

    th {
        background-color: #1976d2;  /* Blue */
        color: white;
    }

    /* Explore More Button */
    .explore-more-btn {
        background-color: #ff9800;  /* Bright Orange */
        color: white;
        padding: 15px 30px;
        font-size: 20px;
        font-weight: bold;
        border-radius: 50px;
        text-align: center;
        cursor: pointer;
        border: none;
        display: block;
        margin-left: auto;
        margin-right: auto;
        animation: pulse 0.5s infinite;
        margin-top: 30px;
    }

    .explore-more-btn:hover {
        background-color: #f57c00;  /* Darker Orange */
        transform: scale(1.05);
    }

    /* Button Style for Visual Appeal */
    .btn {
        padding: 15px 30px;
        font-size: 18px;
        border-radius: 50px;
        text-align: center;
        color: white;
        cursor: pointer;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-top: 10px;
        display: inline-block;
    }

    .btn:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }

    .btn-primary {
        background-color: #4caf50;  /* Green */
    }

    .btn-primary:hover {
        background-color: #388e3c;
    }

    .btn-secondary {
        background-color: #f44336;  /* Red */
    }

    .btn-secondary:hover {
        background-color: #d32f2f;
    }

</style>
""", unsafe_allow_html=True)


# Extract job titles and skills from the CSV file
job_titles = df['Job Title'].sort_values().unique()  # Alphabetical order
skills_dict = dict(zip(df['Job Title'], df['Skills']))

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "About Us", "Resume Analyzer", "Find Jobs", "Enhance Skills", "Contact Us"])

# Header (no line breaks, ensures single-line heading)
st.markdown("<div class='title'>Intelligent Resume Analysis And Job Fit Assessment System</div>", unsafe_allow_html=True)

# Check which page to display using if-else statements
if page == "Home":
    st.markdown("<h1 style='text-align: center;'>Welcome to Our Platform!</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>We help you match your resumes with top job opportunities tailored to your skillset.</p>", unsafe_allow_html=True)
    
    # Path to your logo image (make sure it's correct)
    image_path = r"C:\Users\reeth\Documents\RESUME ANALYZER\images.jpg"  # Adjust the path to your image

    # Open the image using PIL
    image = Image.open(image_path)

    # Display the image in the Streamlit app
    st.image(image,  use_container_width=True)

    # Button to navigate to About Us Page
    # if st.button("Go to About Us"):
    #     st.session_state.page = "About Us"  # Navigate to About Us page


# About Us Page
elif page == "About Us":
    st.markdown("<div class='subtitle'>About Us</div>", unsafe_allow_html=True)
    st.write("""
    # Welcome to the **Intelligent Resume Analysis and Job Fit Assessment System**! 
    Our platform is designed to leverage **Artificial Intelligence** to:
    - Match your resumes with the most relevant job descriptions.
    - Help you discover job opportunities tailored to your skillset.
    - Provide actionable recommendations to enhance your skills.
    
    We aim to make the job search and resume analysis process seamless, accurate, and empowering for job seekers worldwide.
    """)

    # Job Search Dropdown with Skills Display
    st.markdown("<div class='subtitle'>Search for Job Titles</div>", unsafe_allow_html=True)
    search_job_title = st.selectbox("Search Job Title", job_titles)
    if search_job_title:
        st.markdown(f"<div style='margin-left: 20px;'><b>Skills Required for {search_job_title}:</b> {skills_dict[search_job_title]}</div>", unsafe_allow_html=True)

    # Track the index of displayed jobs in session state
    if 'job_index' not in st.session_state:
        st.session_state.job_index = 0  # Start from the first job
        st.session_state.job_list = []  # List to store all displayed jobs

    # Show previously displayed jobs
    st.markdown("<div class='subtitle'>Explore Job Titles</div>", unsafe_allow_html=True)
    for job_title in st.session_state.job_list:
        st.markdown(f"<div style='margin-left: 20px;'><b>{job_title}:</b> {skills_dict[job_title]}</div>", unsafe_allow_html=True)

    # Display jobs in a clickable grid
    end_index = min(st.session_state.job_index + 10, len(job_titles))
    job_subset = job_titles[st.session_state.job_index:end_index]

    col_count = 5 # Set the number of job titles per row
    for i in range(0, len(job_subset), col_count):
        cols = st.columns(col_count)
        for j in range(col_count):
            if i + j < len(job_subset):
                job_title = job_subset[i + j]
                with cols[j]:
                    if st.button(job_title):
                        st.markdown(f"<div style='margin-left: 20px;'><b>Skills Required for {job_title}:</b> {skills_dict[job_title]}</div>", unsafe_allow_html=True)
                        # Add the clicked job to the displayed list
                        if job_title not in st.session_state.job_list:
                            st.session_state.job_list.append(job_title)

    # "Explore more jobs" button with animation
    if end_index < len(job_titles):
        explore_button = st.button("Click here to explore more jobs", key="explore_more", use_container_width=True)
        if explore_button:
            st.session_state.job_index = end_index  # Update to load the next set of jobs


# Resume Analyzer Page

elif page == "Resume Analyzer":
    st.markdown("<div class='subtitle'>Resume Analyzer</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your resume PDF", type="pdf")

    if uploaded_file:
        with st.spinner("Processing resume..."):
            # Extract the text from the PDF
            resume_text = extract_text_from_pdf(uploaded_file)

            # Clean the text to remove extra spaces, newlines, and unwanted characters
            cleaned_resume = resume_text.replace("\n", " ").replace("\r", "").strip()

            # Define a list of keywords related to skills sections, ensuring flexibility
            skill_keywords = ["skills", "technical skills", "technologies", "skills set", "core competencies", "expertise","soft skills", "Programming Languages "]

            # Check if any of the skill keywords are present in the resume
            if not any(re.search(r"\b" + keyword + r"\b", cleaned_resume.lower()) for keyword in skill_keywords):
                st.markdown("No 'Skills' section found in your resume. Please make sure your resume contains a skills section.")
            else:
                # Vectorize resume text
                resume_vector = vectorizer.transform([cleaned_resume])

                # Find the Top 5 Matching Jobs
                distances, indices = knn.kneighbors(resume_vector)

                # Ensure we're always getting the top 5 jobs, even if fewer are found
                num_jobs = 5  # Always fetch 5 jobs

                # Get job titles and their skills
                top_5_jobs = df.iloc[indices[0][:num_jobs]]  # Slice to get only the top 5 jobs
                accuracy_scores = []  # Initialize accuracy_scores list
                job_titles = []  # Initialize a list to store job titles

                # Display the top jobs and calculate accuracy, only if skills are present in the resume
                st.markdown("<div class='subtitle'>Top Matching Job Titles</div>", unsafe_allow_html=True)

                for i in range(num_jobs):
                    job_index = indices[0][i]  # Get the job index
                    score = 1 - distances[0][i]  # Calculate accuracy (1 - distance gives similarity score)

                    job_row = df.iloc[job_index]  # Get the job details using the index

                    # Check if any of the job skills are present in the resume
                    job_skills = job_row['Skills'].lower().split(", ")  # Split the skills into a list
                    if any(skill.lower() in cleaned_resume for skill in job_skills):  # Check if skill is in resume
                        accuracy_scores.append(score)  # Append the score to accuracy_scores
                        job_titles.append(job_row['Job Title'])  # Append the job title

                        # Display job details
                        st.markdown(f"""
                        <div style="
                            border: 5px solid #000;  /* Black border for separation */
                            padding: 15px;
                            margin: 10px 0;
                            border-radius: 10px;
                            font-size: 1.1em;
                        ">
                            <strong>Job Title:</strong> {job_row['Job Title']}<br>
                            <strong>Matched Skills:</strong> {job_row['Skills']}<br>
                            <strong>Accuracy:</strong> {score:.2f}
                        </div>
                        """, unsafe_allow_html=True)

                # If no jobs match, display a message
                if not accuracy_scores:
                    st.markdown("No matching job titles found based on the skills in your resume.")

                # Only display the "Top Matching Job" message if jobs were found
                if accuracy_scores:
                    # Sort the accuracy scores and job titles together
                    sorted_jobs = sorted(zip(accuracy_scores, job_titles), reverse=True)

                    # Get the top matched job
                    top_job_name = sorted_jobs[0][1]

                    st.markdown(f"""
                    <div style="
                        font-size: 2em;
                        font-weight: bold;
                        color: #ff6347;  /* Tomato color for emphasis */
                        text-align: center;
                        animation: pulse 2s infinite;
                    ">
                        Top Matching Job: <span style="color: #008080;">{top_job_name}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # CSS animation for highlighting the job title
                    st.markdown(f"""
                    <style>
                        @keyframes pulse {{
                            0% {{ transform: scale(1); }}
                            50% {{ transform: scale(1.1); }}
                            100% {{ transform: scale(1); }}
                        }}
                    </style>
                    """, unsafe_allow_html=True)

                    # Pie chart visualization for the top job accuracy scores
                    if accuracy_scores:
                        # Plot the pie chart
                        colors = plt.cm.Paired.colors  # Color palette
                        fig, ax = plt.subplots()
                        ax.pie(accuracy_scores, labels=job_titles, autopct='%1.1f%%', startangle=90, colors=colors)
                        ax.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.

                        st.pyplot(fig)

                    # Display the "Keep it up!" motivational message after the chart
                    st.markdown(""" 
                    <div class='subtitle' style="color:green;">Keep it up! You're on the right track to finding your dream job!</div>
                    <p style="text-align:center;">By analyzing your resume, we've matched you with top roles based on your skills fit. Keep enhancing your skills and applying for opportunities!</p>
                    """, unsafe_allow_html=True)
                else:
                    # Display the "No matching job found" message when no accuracy scores
                    st.markdown(""" 
                    <div class='subtitle' style="color:red;">No matching job found based on your resume.</div>
                    """, unsafe_allow_html=True)


# Find Jobs Section
if page == "Find Jobs":
    
    st.markdown("""
    <div class='content'>
        <h3 class='subtitle'>Welcome to the Find Jobs Page!</h3>
        <ul>
            <li>Explore various job opportunities from top portals.</li>
            <li>Whether you are looking for internships, part-time jobs, or full-time positions, these platforms have a wide range of listings.</li>
            <li>These platforms cater to different skill sets and interests.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='content'>
        <h3 class='subtitle'>Take Your Time to Explore</h3>
        <ul>
            <li>Take your time to browse through the available positions, and apply to the ones that align with your career goals.</li>
            <li>Every application is a step closer to your dream job.</li>
            <li>Keep learning, keep growing, and stay motivated!</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='content'>
        <h3 class='subtitle'>Job Opportunities From Top Portals</h3>
        <ul>
            <li>Here are some top portals where you can find exciting job opportunities:</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    # Custom CSS for styling the page
    st.markdown("""
    <style>
   
    
   
    
    .job-portal-link {
        background-color: #b0bec5;
        color: black;
        padding: 12px 20px;
        border-radius: 8px;
        font-size: 18px;
        display: inline-block;
        text-decoration: none;
        margin: 10px 0;
        width: 100%;
        text-align: center;
    }
    
    .job-portal-link:hover {
        background-color: white;
    }
    
    .columns {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
    }
    
    .column {
        width: 30%;
        margin-bottom: 15px;
    }
    
    .encouragement {
        font-size: 20px;
        font-weight: bold;
        color: blue;
        padding: 20px;
        border-radius: 10px;
        margin-top: 30px;
    }
    
    .tip {
        font-size: 18px;
        color: #16A085;
        font-style: italic;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

   
    
    # Create 3 columns for the links
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<a href="https://www.unstop.com/jobs" target="_blank" class="job-portal-link">Find Jobs on Unstop</a>', unsafe_allow_html=True)
        st.markdown('<a href="https://www.workindia.in/jobs" target="_blank" class="job-portal-link">Find Jobs on WorkIndia</a>', unsafe_allow_html=True)
        st.markdown('<a href="https://www.internshala.com/internships" target="_blank" class="job-portal-link">Find Internships on Internshala</a>', unsafe_allow_html=True)

    with col2:
        st.markdown('<a href="https://www.linkedin.com/jobs" target="_blank" class="job-portal-link">Find Jobs on LinkedIn</a>', unsafe_allow_html=True)
        st.markdown('<a href="https://www.glassdoor.com/Job/index.htm" target="_blank" class="job-portal-link">Find Jobs on Glassdoor</a>', unsafe_allow_html=True)
        st.markdown('<a href="https://www.indeed.com" target="_blank" class="job-portal-link">Find Jobs on Indeed</a>', unsafe_allow_html=True)

    with col3:
        st.markdown('<a href="https://www.naukri.com" target="_blank" class="job-portal-link">Find Jobs on Naukri</a>', unsafe_allow_html=True)
        st.markdown('<a href="https://www.angel.co/jobs" target="_blank" class="job-portal-link">Find Jobs on AngelList</a>', unsafe_allow_html=True)
        st.markdown('<a href="https://www.simplyhired.com" target="_blank" class="job-portal-link">Find Jobs on SimplyHired</a>', unsafe_allow_html=True)

    # Additional Encouragement
    st.markdown("<div class='encouragement'>Pro Tip: While applying, make sure to tailor your resume for each job. Highlight relevant skills, experiences, and achievements that align with the job description.</div>", unsafe_allow_html=True)

    st.markdown("<div class='encouragement'>Keep improving your skills and learning new ones. The right job is just around the corner!</div>", unsafe_allow_html=True)

# Enhance Skills Page
if page == "Enhance Skills":
    # Introduction Section
    st.markdown("<div ><h3 class='subtitle'>Enhance Your Skills for Success</h3></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='content'>
        <p>Continuous learning and skill development are essential for staying competitive in the job market. Here, you can find a variety of resources to help you improve your skills, whether you're looking to boost your technical abilities or enhance your soft skills.</p>
    </div>
    """, unsafe_allow_html=True)

    # Top Skills to Learn
    st.markdown("<div ><h3 class='subtitle'>Top Skills to Learn</h3></div>", unsafe_allow_html=True)
    st.markdown("""
    <ul>
        <li><strong>Technical Skills:</strong> Data Analysis, Programming (Python, JavaScript), Machine Learning, Cloud Computing</li>
        <li><strong>Soft Skills:</strong> Communication, Leadership, Problem-Solving, Time Management</li>
        <li><strong>Industry-Specific Skills:</strong> Digital Marketing, UX/UI Design, Financial Analysis, Business Development</li>
    </ul>
    """, unsafe_allow_html=True)

    # Online Courses & Certifications
    st.markdown("<div '><h3 class='subtitle'>Recommended Courses & Certifications</h3></div>", unsafe_allow_html=True)
    
    # Creating 3 columns for links to be grouped
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<a href="https://www.coursera.org" target="_blank" class="skill-button">Coursera </a>', unsafe_allow_html=True)
        st.markdown('<a href="https://www.edx.org" target="_blank" class="skill-button">edX</a>', unsafe_allow_html=True)
        st.markdown('<a href="https://www.udemy.com" target="_blank" class="skill-button">Udemy</a>', unsafe_allow_html=True)

    with col2:
        st.markdown('<a href="https://www.linkedin.com/learning" target="_blank" class="skill-button">LinkedIn</a>', unsafe_allow_html=True)
        st.markdown('<a href="https://www.skillshare.com" target="_blank" class="skill-button">Skillshare </a>', unsafe_allow_html=True)
        st.markdown('<a href="https://www.futurelearn.com" target="_blank" class="skill-button">FutureLearn</a>', unsafe_allow_html=True)

    with col3:
        st.markdown('<a href="https://www.codecademy.com" target="_blank" class="skill-button">Codecademy </a>', unsafe_allow_html=True)
        st.markdown('<a href="https://www.leetcode.com" target="_blank" class="skill-button">LeetCode </a>', unsafe_allow_html=True)
        st.markdown('<a href="https://www.khanacademy.org" target="_blank" class="skill-button">Khan Academy</a>', unsafe_allow_html=True)

    # Skill-Building Tools & Platforms
    st.markdown("<div><h3 class='subtitle'>Skill-Building Tools & Platforms</h3></div>", unsafe_allow_html=True)
    st.markdown("""
    <ul>
        <li><a href="https://www.codecademy.com" target="_blank" class="skill-button">Codecademy - Learn programming interactively</a></li>
        <li><a href="https://www.leetcode.com" target="_blank" class="skill-button">LeetCode - Practice coding and algorithms</a></li>
        <li><a href="https://www.duolingo.com" target="_blank" class="skill-button">Duolingo - Learn languages in a fun way</a></li>
    </ul>
    """, unsafe_allow_html=True)

    # Tips for Skill Building
    st.markdown("<div class='subtitle'><h3 class='subtitle'>Tips for Effective Skill Building</h3></div>", unsafe_allow_html=True)
    st.markdown("""
    <ul>
        <li>Start with small, achievable goals to stay motivated.</li>
        <li>Consistency is key. Dedicate time daily or weekly for practice.</li>
        <li>Take online courses to gain structured knowledge and certifications.</li>
        <li>Join communities or forums to network and learn from others.</li>
    </ul>
    """, unsafe_allow_html=True)

    # Interactive Challenges Section
    st.markdown("<div class='subtitle'><h3 class='subtitle'>Take On Challenges</h3></div>", unsafe_allow_html=True)
    st.markdown("""
    <ul>
        <li>Participate in coding challenges on <a href="https://www.hackerrank.com" target="_blank" >HackerRank</a> or <a href="https://www.codewars.com" target="_blank" >Codewars</a>.</li>
        <li>Join design challenges on <a href="https://dribbble.com" target="_blank" >Dribbble</a> to improve your creative skills.</li>
    </ul>
    """, unsafe_allow_html=True)

    # Job-Relevant Skill Sets
    
    # Add CSS Styling for Hover Effects and Button Styling
    st.markdown("""
    <style>
    .skill-button {
        background-color: #b0bec5;
        color: blue;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        border-radius: 8px;
        transition: background-color 0.3s ease;
    }
    .skill-button:hover {
        background-color: white;
    }
    
    .content {
        font-size: 18px;
        line-height: 1.6;
    }
    
    .content h3 {
        color: #333;
        font-size: 24px;
    }
    </style>
    """, unsafe_allow_html=True)


# Contact Us Page
# import mysql.connector
# import streamlit as st

# Page selection


# Function to save data to MySQL
# def save_to_mysql(name, email, message, phone, rating):
    # try:
    #     # Connect to MySQL
    #     conn = mysql.connector.connect(
    #         host="localhost",    # MySQL server host
    #         user="root",         # MySQL username
    #         password="IndiraKedila",  # MySQL password
    #         database="contact_form",
    #         charset="utf8mb4" 
    #           # Database name
    #     )
        
    #     cursor = conn.cursor()

    #     # Insert data into "messages" table
    #     cursor.execute('''
    #         INSERT INTO messages (name, email, message, phone, rating)
    #         VALUES (%s, %s, %s, %s, %s)
    #     ''', (name, email, message, phone, rating))

    #     conn.commit()  # Commit the transaction
    #     cursor.close()
    #     conn.close()

    # except mysql.connector.Error as err:
    #     st.error(f"Error occurred: {err}")

# Streamlit page for Contact Us
if page == "Contact Us":
    st.markdown("<div class='subtitle'>Contact Us</div>", unsafe_allow_html=True)

    # Description for the form
    st.write("""
    **We'd love to hear from you!**
    If you have any questions, feedback, or need assistance, feel free to reach out to us.
    You can contact us using the following methods:

    - **Email**: [resumeanalyzerr@gmail.com](mailto:resumeanalyzerr@gmail.com)
    - **Phone**: +91 7676346378
    """)

    # Contact Form (Improved design)
    with st.form(key="contact_form", clear_on_submit=True):
        # Fields for Name, Email, Phone, and Message
        contact_name = st.text_input("Your Name", max_chars=50)
        contact_email = st.text_input("Your Email", max_chars=100)
        contact_phone = st.text_input("Your Phone Number", max_chars=15)
        contact_message = st.text_area("Your Message", max_chars=500, height=150)

        # Star Rating using custom HTML (can be customized further)
        emojis = ["😡", "😞", "😐", "😊", "😍"]
        rating = st.radio("Rate Us", emojis, index=2, horizontal=True)
        st.write(f"Rating selected: {rating}")  # Debugging line to verify the emoji
        # Submit button
        submit_button = st.form_submit_button("Submit")

        if submit_button:
            # Check if all fields are filled
            if contact_name.strip() and contact_email.strip() and contact_message.strip() and contact_phone.strip():
                # Save to MySQL
                # save_to_mysql(contact_name.strip(), contact_email.strip(), contact_message.strip(), contact_phone.strip(), rating)
                st.success("Thank you for your feedback! We'll get back to you shortly.")
            else:
                st.error("Please fill out all fields before submitting.")

    # Custom CSS for better design and layout
    st.markdown("""
    <style>    
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border: none;
        cursor: pointer;
        padding: 10px 20px;
        border-radius: 5px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }

    .stButton>button:hover {
        background-color: #45a049;
    }

    /* Input fields */
    .stTextInput>div>input {
        font-size: 16px;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ccc;
    }

    .stTextArea>div>textarea {
        font-size: 16px;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ccc;
    }

    /* Text alignment */
    .stTextInput, .stTextArea {
        margin-bottom: 20px;
    }

    /* Rating Style (Stars) */
    .stRadio>div>label>div {
        display: flex;
        justify-content: center;
        font-size: 24px;
        color: #FFD700;  /* Gold color for stars */
    }

    .stRadio>div>label>div>input {
        cursor: pointer;
    }
    
    </style>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>© 2024 Resume Analyzer (ARKK)</div>", unsafe_allow_html=True)






# import pandas as pd
# import numpy as np
# import re
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.neighbors import NearestNeighbors
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# import fitz  # PyMuPDF for PDF handling
# import streamlit as st
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# import csv
# import sqlite3
# import getpass  # For secure password input

# # Download necessary nltk resources
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

# st.markdown("""
#     <style>
#         body {
#             background-color: #ADD8E6;
#            color: #000000;
#        }
#               .title {
#             text-align: center;
#             color: #4a90e2;
#             font-size: 48px;
#             font-weight: bold;
#            padding: 10px;#         }
#          .subtitle {
#              text-align: center;
#             font-size: 24px;
#              color: #000000;
#            margin-top: -10px;
#        }
#          .result {
#              border: 2px solid #4a90e2;
#              border-radius: 10px;
#              padding: 20px;
#              background-color: #f1f1f1;
#             margin-bottom: 10px;
#          }
#          .footer {
#              text-align: center;
#              font-size: 14px;
#              color: #888;
#             padding-top: 20px;
#          }
#      </style>
#  """, unsafe_allow_html=True)

# # Read dataset
# df = pd.read_csv('cleaned_file.csv')

# # Clean the job title and skills
# def clean_text(txt):
#     clean_text = re.sub('http\S+\s', ' ', txt)
#     clean_text = re.sub('RT|cc', ' ', clean_text)
#     clean_text = re.sub('#\S+\s', ' ', clean_text)
#     clean_text = re.sub('@\S+', '  ', clean_text)
#     clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
#     clean_text = re.sub(r'[^\x00-\x7f]', ' ', clean_text)
#     clean_text = re.sub('\s+', ' ', clean_text)
#     tokens = nltk.word_tokenize(clean_text.lower())
#     stop_words = set(stopwords.words('english'))
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalpha()]
#     return ' '.join(tokens)

# # Ensure 'cleaned_job_title' is created
# if 'Job Title' in df.columns:
#     df['cleaned_job_title'] = df['Job Title'].apply(clean_text)
# else:
#     print("Error: 'Job Title' column not found in the dataset.")
#     exit()

# # Ensure 'Skills' column exists and clean it if necessary
# if 'Skills' in df.columns:
#     df['cleaned_skills'] = df['Skills'].apply(clean_text)
# else:
#     print("Error: 'Skills' column not found in the dataset.")
#     exit()

# # Define relevant keywords and filter the data
# relevant_keywords = ['engineer', 'developer', 'data', 'machine learning', 'cloud', 'AI', 'IoT']

# def is_relevant_job(title):
#     return any(keyword in title for keyword in relevant_keywords) and ('marine' not in title)

# df['relevant'] = df['cleaned_job_title'].apply(is_relevant_job)
# df = df[df['relevant']]  # Filter the dataframe to keep only relevant rows

# # Create the job_description column
# df['job_description'] = df['cleaned_job_title'] + ' ' + df['cleaned_skills']

# # Function to clean resume text
# def clean_resume(txt):
#     return clean_text(txt)

# # Function to extract text from PDF
# def extract_text_from_pdf(pdf_file):
#     try:
#         text = ""
#         with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
#             for page in doc:
#                 text += page.get_text()
#         return text
#     except Exception as e:
#         st.error(f"Failed to process the PDF file: {e}")
#         return ""

# # Function to send email
# def send_email(name, sender_email, message):
#     receiver_email = "4mt21ic039@mite.ac.in"  # This is your email where you receive the messages
#     sender_password = "Kedila@1975"  # This is the app-specific password for your email
    
#     msg = MIMEMultipart()
#     msg['From'] = sender_email  # User's email entered in the form
#     msg['To'] = receiver_email  # Your email (receiver)
#     msg['Subject'] = "New Contact Form Submission"
    
#     body = f"Name: {name}\nEmail: {sender_email}\nMessage: {message}"
#     msg.attach(MIMEText(body, 'plain'))

#     try:
#         # Use SSL for a secure connection
#         server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
#         server.login("4mt21ic039@mite.ac.in", sender_password)  # Log in with your email and app password
#         server.sendmail(sender_email, receiver_email, msg.as_string())  # Send the email
#         server.quit()  # Close the connection

#         print(f"Email sent successfully to {receiver_email}")
#     except Exception as e:
#         print(f"Failed to send email: {e}")

# # Function to save contact data to a CSV file
# def save_to_file(name, email, message):
#     with open("contact_submissions.csv", "a", newline="") as file:
#         writer = csv.writer(file)
#         writer.writerow([name, email, message])
#     st.success(f"Thank you for contacting us, {name}! We will get back to you soon.")

# # Function to save contact data to SQLite
# def save_to_db(name, email, message):
#     conn = sqlite3.connect('contact_form.db')
#     c = conn.cursor()
#     c.execute('CREATE TABLE IF NOT EXISTS contacts (name TEXT, email TEXT, message TEXT)')
#     c.execute('INSERT INTO contacts (name, email, message) VALUES (?, ?, ?)', (name, email, message))
#     conn.commit()
#     conn.close()
#     # st.success(f"Thank you for contacting us, {name}! We will get back to you soon.")

# # Streamlit app content
# st.markdown("<div class='title'>Resume Analyzer</div>", unsafe_allow_html=True)

# # Add navigation options
# navigation = st.sidebar.selectbox("Navigation", ["Home", "About Us", "Search for Job Title", "Explore Job Titles", "Apply for Matched Job", "Contact Us"])

# # About Us section
# if navigation == "About Us":
#     st.title("About Us")
#     st.write("""**Resume Analyzer** is a tool designed to assist job seekers in matching their resumes to the right job titles.""")

# # Job Title Search
# elif navigation == "Search for Job Title":
#     st.title("Search for Job Title")
#     selected_job_title = st.selectbox("Select a Job Title", df['Job Title'].unique())
#     if selected_job_title:
#         job_data = df[df['Job Title'] == selected_job_title]
#         st.write(f"**Job Title:** {selected_job_title}")
#         st.write("**Matching Skills:**")
    
#     for _, row in job_data.iterrows():
#         skills_list = row['cleaned_skills'].split()  # Split skills into a list by space
#         skills_text = ', '.join(skills_list)  # Join them with commas
#         st.write(skills_text)

# # Explore Job Titles section
# elif navigation == "Explore Job Titles":
#     st.title("Explore Job Titles")
    
#     job_titles = df['Job Title'].unique()
#     columns = st.columns(3)  # 3 columns layout, adjust if necessary
    
#     selected_job = None
#     for i, job_title in enumerate(job_titles):
#         with columns[i % 3]:  # Loop through columns and arrange in a grid
#             if st.button(job_title):
#                 selected_job = job_title
#                 break
    
#     if selected_job:
#         job_data = df[df['Job Title'] == selected_job]
#         st.write(f"**Job Title:** {selected_job}")
#         st.write("**Matching Skills:**")
        
#         for _, row in job_data.iterrows():
#             skills_list = row['cleaned_skills'].split()  # Split skills into a list by space
#             skills_text = ', '.join(skills_list)  # Join them with commas
#             st.write(skills_text)

# # Apply for Matched Job section
# elif navigation == "Apply for Matched Job":
#     st.title("Apply for Matched Job")
#     st.markdown("<h3 style='color: #1f77b4;'>Click Here to Find Multiple Job Opportunities</h3>", unsafe_allow_html=True)
#     job_portal_url = "https://www.googleadservices.com/pagead/aclk?sa=L&ai=DChcSEwjknoXNmf-JAxXWpWYCHXYeAS4YABAAGgJzbQ"  # Replace with the actual job portal link
#     st.markdown(f"[Click here to explore and apply for multiple jobs on the portal]({job_portal_url})", unsafe_allow_html=True)

# # Contact Us section
# elif navigation == "Contact Us":
#     st.title("Contact Us")
    
#     # Contact form fields
#     name = st.text_input("Your Name")
#     email = st.text_input("Your Email")
#     message = st.text_area("Your Message")
    
#     # Button to submit the form
#     if st.button("Submit"):
#         if name and email and message:
#             # Save to database, file, and send email
#             save_to_file(name, email, message)
#             save_to_db(name, email, message)
#             send_email(name, email, message)
#         else:
#             st.error("Please fill in all the fields.")

# # Home Page
# elif navigation == "Home":
#     st.title("Welcome to Resume Analyzer")
#     st.markdown("<div class='subtitle'>Upload your resume to find matching job titles</div>", unsafe_allow_html=True)
#     uploaded_file = st.file_uploader("Upload a resume PDF", type="pdf")

#     if uploaded_file:
#         with st.spinner("Processing resume..."):
#             resume_text = extract_text_from_pdf(uploaded_file)
#             if not resume_text.strip():
#                 st.error("No text found in the uploaded PDF.")
#                 st.stop()
#             cleaned_resume = clean_resume(resume_text)
#             vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8)
#             job_vectors = vectorizer.fit_transform(df['job_description']).toarray()
#             knn = NearestNeighbors(n_neighbors=5, metric='cosine')
#             knn.fit(job_vectors)
#             resume_vector = vectorizer.transform([cleaned_resume]).toarray()
#             distances, indices = knn.kneighbors(resume_vector)
#             top_5_jobs = df.iloc[indices[0]]
            
#             job_titles = top_5_jobs['Job Title']
#         skills_count = [len(row['cleaned_skills'].split()) for _, row in top_5_jobs.iterrows()]

#         st.subheader("Number of Skills Matched with Top 5 Jobs")
#         fig, ax = plt.subplots(figsize=(10, 6))
#         sns.barplot(x=job_titles, y=skills_count, palette='viridis', ax=ax)
#         ax.set_title('Top 5 Job Matches for the Resume')
#         ax.set_xlabel('Job Titles')
#         ax.set_ylabel('Number of Skills Matched')

#         # Rotate the job titles
#         plt.xticks(rotation=45, ha='right')

#         st.pyplot(fig)

# else:
#     st.warning("Please upload a resume to proceed.")

# # Footer
# st.markdown("<div class='footer'>© 2024 Resume Analyzer (ARKK)</div>", unsafe_allow_html=True)





# import streamlit as st
# import pandas as pd
# import numpy as np
# import re
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.neighbors import NearestNeighbors
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# import fitz  # PyMuPDF for PDF handling

# # Download necessary NLTK resources
# nltk.download('punkt')  # Standard Punkt tokenizer for word tokenization
# nltk.download('stopwords')  # Stopwords for text cleaning
# nltk.download('wordnet')  # WordNet for lemmatization

# # Load the Dataset
# try:
#     df = pd.read_csv('cleaned_file.csv')
#     if 'Job Title' not in df.columns or 'Skills' not in df.columns:
#         st.error("Dataset must contain 'Job Title' and 'Skills' columns.")
# except FileNotFoundError:
#     st.error("File 'cleaned_file.csv' not found. Please ensure it is in the same directory as this script.")
#     st.stop()

# # Preprocessing and Cleaning Functions
# def clean_text(txt):
#     if not isinstance(txt, str):
#         return ""
#     stop_words = set(stopwords.words('english'))
#     lemmatizer = WordNetLemmatizer()
#     clean_txt = re.sub(r'http\S+|RT|cc|#\S+|@\S+|[^\x00-\x7f]', ' ', txt)
#     clean_txt = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_txt)
#     clean_txt = re.sub(r'\s+', ' ', clean_txt).strip().lower()
#     tokens = nltk.word_tokenize(clean_txt)  # Tokenize using the Punkt tokenizer
#     tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
#     return ' '.join(tokens)

# # Apply cleaning functions to dataset
# try:
#     df['cleaned_job_title'] = df['Job Title'].apply(clean_text)
#     df['cleaned_skills'] = df['Skills'].apply(clean_text)
#     df['job_description'] = df['cleaned_job_title'] + ' ' + df['cleaned_skills']
# except KeyError:
#     st.error("Error in processing the dataset. Ensure 'Job Title' and 'Skills' columns exist.")
#     st.stop()

# # Vectorization and KNN Model
# try:
#     vectorizer = TfidfVectorizer(max_features=5000)
#     job_vectors = vectorizer.fit_transform(df['job_description']).toarray()

#     knn = NearestNeighbors(n_neighbors=5, metric='cosine')
#     knn.fit(job_vectors)
# except ValueError as e:
#     st.error(f"Error in vectorization or model training: {e}")
#     st.stop()

# # Function to process PDF and return text content
# def extract_text_from_pdf(pdf_file):
#     try:
#         text = ""
#         with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
#             for page in doc:
#                 text += page.get_text()
#         return text
#     except Exception as e:
#         st.error(f"Failed to process the PDF file: {e}")
#         return ""

# # Custom CSS styling
# st.markdown("""
#     <style>
#         body {
#             background-color: #ADD8E6;
#             color: #000000;
#         }
#         .title {
#             text-align: center;
#             color: #4a90e2;
#             font-size: 48px;
#             font-weight: bold;
#             padding: 10px;
#         }
#         .subtitle {
#             text-align: center;
#             font-size: 24px;
#             color: #000000;
#             margin-top: -10px;
#         }
#         .result {
#             border: 2px solid #4a90e2;
#             border-radius: 10px;
#             padding: 20px;
#             background-color: #f1f1f1;
#             margin-bottom: 10px;
#         }
#         .footer {
#             text-align: center;
#             font-size: 14px;
#             color: #888;
#             padding-top: 20px;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # Streamlit App Layout with HTML structure
# st.markdown("<div class='title'>Resume Analyzer</div>", unsafe_allow_html=True)
# st.markdown("<div class='subtitle'>Upload your resume to find matching job titles</div>", unsafe_allow_html=True)

# # File uploader
# uploaded_file = st.file_uploader("Upload a resume PDF", type="pdf")

# if uploaded_file:
#     try:
#         with st.spinner("Processing resume..."):
#             resume_text = extract_text_from_pdf(uploaded_file)
#             if not resume_text.strip():
#                 st.error("No text found in the uploaded PDF.")
#                 st.stop()

#             cleaned_resume = clean_text(resume_text)

#         # Vectorize resume text
#         resume_vector = vectorizer.transform([cleaned_resume]).toarray()

#         # Find the Top 5 Matching Jobs
#         distances, indices = knn.kneighbors(resume_vector)
#         top_5_jobs = df.iloc[indices[0]]

#         st.markdown("<div class='subtitle'>Top 5 Matching Job Titles</div>", unsafe_allow_html=True)

#         for i, row in top_5_jobs.iterrows():
#             st.markdown(
#                 f"<div class='result'><b>Job Title:</b> {row['Job Title']}<br><b>Matched Skills:</b> {row['Skills']}</div>",
#                 unsafe_allow_html=True
#             )

#         # Visualization
#         job_titles = top_5_jobs['Job Title']
#         skills_count = [len(row['cleaned_skills'].split()) for _, row in top_5_jobs.iterrows()]

#         st.subheader("Number of Skills Matched with Top 5 Jobs")
#         fig, ax = plt.subplots(figsize=(10, 6))
#         sns.barplot(x=job_titles, y=skills_count, palette='viridis', ax=ax)
#         ax.set_title('Top 5 Job Matches for the Resume')
#         ax.set_xlabel('Job Titles')
#         ax.set_ylabel('Number of Skills Matched')

#         # Rotate the job titles
#         plt.xticks(rotation=45, ha='right')

#         st.pyplot(fig)

#     except Exception as e:
#         st.error(f"An error occurred while processing the file: {e}")

# # Footer
# st.markdown("<div class='footer'>© 2024 Resume Analyzer (ARKK)</div>", unsafe_allow_html=True)
