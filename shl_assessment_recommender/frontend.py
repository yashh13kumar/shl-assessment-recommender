import streamlit as st
import requests

API_URL = "https://shl-assessment-recommender-yj81.onrender.com"

st.title("SHL Assessment Recommendation System")

query = st.text_area("Enter a job description, natural language query, or URL:")

if st.button("Get Recommendations"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Fetching recommendations..."):
            try:
                response = requests.post(f"{API_URL}/recommend", json={"query": query})
                response.raise_for_status()
                data = response.json()
                recommendations = data.get("recommendations", [])
                if recommendations:
                    st.write(f"Top {len(recommendations)} Recommendations:")
                    st.table([
                        {
                            "Assessment Name": r["name"],
                            "URL": r["url"],
                            "Remote Testing": r["remote_testing"],
                            "Adaptive/IRT": r["adaptive_irt"],
                            "Duration": r["duration"],
                            "Test Type": r["test_type"]
                        }
                        for r in recommendations
                    ])
                else:
                    st.info("No recommendations found.")
            except Exception as e:
                st.error(f"Error fetching recommendations: {e}")
