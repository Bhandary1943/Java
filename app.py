import streamlit as st
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import fitz  # PyMuPDF for PDF handling

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Predefined relevant keywords for job filtering
relevant_keywords = ['iot', 'cybersecurity', 'machine learning', 'ai', 'data science', 'blockchain', 
                     'developer', 'engineer', 'software', 'embedded', 'technologist']

# Define the cleaning function for text data
def clean_text(txt):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    clean_text = re.sub(r'http\S+\s|RT|cc|#\S+\s|@\S+|[^\x00-\x7f]', ' ', str(txt))
    clean_text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip().lower()
    tokens = word_tokenize(clean_text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalpha()]
    return ' '.join(tokens)

df = pd.DataFrame()
# Handle File Upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    try:
        # Read the uploaded CSV file into a DataFrame
        df = pd.read_csv('cleaned_file.csv')
        
        # Debugging: Display the columns of the uploaded file
        st.write("Columns in the uploaded CSV file:", df.columns)

        # Check if the necessary columns ('Job Title' and 'Skills') are in the DataFrame
        if 'Job Title' not in df.columns or 'Skills' not in df.columns:
            st.error("The required columns 'Job Title' or 'Skills' are missing in the uploaded file.")
        else:
            st.success("File uploaded successfully")
            
            # Handle missing values (if any) in 'Job Title' or 'Skills' columns
            df = df.dropna(subset=['Job Title', 'Skills'])

            # Apply cleaning function to job title and skills
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

    except Exception as e:
        st.error(f"Error occurred while processing the file: {str(e)}")

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
if not df.empty:
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
