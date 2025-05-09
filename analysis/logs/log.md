## Prompts
Following the [Rose/Bud/Thorn](https://www.panoramaed.com/blog/rose-bud-thorn-activity-and-worksheet#:~:text=%22Rose%2C%20Bud%2C%20Thorn%22%20is%20a%20mindful%20design%2D,day%2C%20week%2C%20or%20month.) model:

### Date: 
Week number, today's date, etc. 
### Number of hours: 
A quantity of hours, maybe towards specific tasks. 
### Rose:
The highlight from the previous weekly/bi-weekly working period, such as something you found particularly rewarding. This could also be something you're excited to implement now.
### Bud: 
Something that you are looking forward to digging into deeper. This could also be ideas on how to apply concepts to your research in the future. 
### Thorn: 
Something that was challenging that could be worked on, such as anything that wasn't 100% clear and could be elaborated on. Any sticking points should be addressed here. 
### Additional thought
Write anything that you think would be important for YOU later on.

------------------------------------------------------------------------------------------------

# Weekly/Bi-Weekly Log (Jose F Oviedo)

### Date: 2025-03-12
### Number of hours: 2.5 hours
### Rose: Got to experiment with APIs and web scraping.
### Bud: How to use LMs to organize unstructured data.
### Thorn: Cleaning the scraped html data for better model understanding.
### Additional thoughts: Try different frameworks until you find the right one for the job.

### Date: 2025-03-24
### Number of hours: 1.5 hours
### Rose: Getting LMs to process our documents and give us the structured outputs we want.
### Bud: There are different ways to get a model to process unstructured data depending on the model capabilities.
### Thorn: There are so many models available with different strengths and weaknesses.
### Additional thoughts: Test different models and compare outputs.

### Date: 2025-03-31
### Number of hours: 2 hours
### Rose: regenerated structured outputs for simpler schema better suited to our vector database.
### Bud: We are moving on to the next step of developing the application systems.
### Thorn: The original structured outputs were too nested for our application. 
### Additional thoughts: review system constraints frequently.

### Date: 2025-04-03
### Number of hours: 2 hours
### Rose: Put together the proposal with the team and sent to mentors.
### Bud: We will take suggestions from team mentors and implement them. 
### Thorn: proposal revisions.
### Additional thoughts: Work with team to stay on track and reach milestones.

### Date: 2025-04-25
### Number of hours: 4 hours
### Rose: Coded a basic vector db retriever for testing the db.
### Bud: We will move on to implementing a full rag system connected to an LLM. 
### Thorn: We plan on using Google Gemini LMs but have had compatibility issues.
### Additional thoughts: Work with team to maintain communication. 

### Date: 2025-05-01
### Number of hours: 3 hours
### Rose: Implemented a RAG system that connects to our FAISS vector db.
### Bud: We will deploy webapp to cloud with streamlit for uiux plase.  
### Thorn: The RAG system that worked most robustly ended up using openai LMs and langchain instead of Google LMs.
### Additional thoughts: Work with team to maintain communication. 

### Date: 2025-05-02
### Number of hours: 1 hours
### Rose: Team put forward poster content and we are reviewing it before finalizing poster.
### Bud: We have a good amount of poster content to work with and also a plan for our app demo using printed publications that were digested into the KB.
### Thorn: We havent yet connected the uiux to the backend RAG system but are working on it.
### Additional thoughts: Choose final poster design with the team. 

------------------------------------------------------------------------------------------------


# Weekly/Bi-Weekly Log (Visalakshi Prakash Iyer)

### Date: 
02/08/2025
### Number of hours: 
3
### Rose:
Successfully organized and led the requirement definition meeting. It was rewarding to see the team actively contribute ideas, which I synthesized into a comprehensive flowchart outlining the application's architecture. This collaborative effort clarified our vision and set a strong foundation for the project.
### Bud: 
I'm looking forward to refining the requirements further and translating our high-level ideas into actionable tasks. I'm also excited to explore how we can leverage the latest advancements in vector databases and LLM integration for our use case.
### Thorn: 
Coordinating everyone's availability for the initial meeting was a challenge, and some requirements were still a bit vague after the first session. I need to ensure we revisit these points and get explicit agreement on scope to avoid confusion later. 
### Additional thoughts:
Documenting early decisions in detail will be crucial. I should keep a running list of open questions to revisit in future meetings to ensure nothing falls through the cracks.

### Date: 
02/16/2025
### Number of hours: 
8
### Rose:
I compiled the final architecture and led several productive brainstorming sessions. It was satisfying to troubleshoot gaps in our feature designs and finalize the essential tech stack. The team’s engagement and willingness to iterate on ideas made this phase particularly successful.
### Bud: 
Looking forward to seeing how our chosen architecture performs in practice, especially the integration between the LLM and the vector database. I’m eager to start prototyping and validating our design decisions.
### Thorn: 
Some technical trade-offs were difficult, especially regarding scalability and data flow. I want to revisit these decisions after initial implementation to ensure they hold up under real-world conditions.
### Additional thoughts:
NA

### Date: 
03/12/2025
### Number of hours: 
3
### Rose:
Provided the starter path for implementation of the vector database and oversaw the data transformation structure. It was rewarding to see the team make progress on the backend, and my guidance helped avoid early pitfalls.
### Bud: 
Looking forward to deeper testing and optimization of the vectorstore, and to integrating it with the LLM for real-time knowledge retrieval.
### Thorn: 
Some initial confusion around the data transformation pipeline required extra troubleshooting.
### Additional thoughts:
NA

### Date: 
03/23/2025
### Number of hours: 
10
### Rose:
Led the team through the proposal writing and submission process. It was gratifying to see our collective ideas formalized into a coherent document and receive positive feedback from stakeholders.
### Bud: 
Excited to transition from planning to implementation and start building the core components. Looking forward to seeing our ideas take shape in code.
### Thorn: 
NA
### Additional thoughts:
Keep a template for future proposals and note what worked well in this process to streamline future submissions. 

### Date: 
04/12/2025
### Number of hours: 
6
### Rose:
Successfully tested the implementation of the vectorstore. It was satisfying to validate that our design choices worked as expected and to see the system retrieving relevant information accurately.
### Bud: 
Looking forward to integrating the UI with the backend and seeing the full workflow in action. Eager to gather user feedback on the system’s performance.
### Thorn: 
Encountered low semantic similarity score for some queries, that need to be dealth with
### Additional thoughts:
Maybe we can change the way we split the chunks of data to see if the score increases/decreases

### Date: 
04/20/2025 - 05/03/2025
### Number of hours: 
40-42
### Rose:
I developed and implemented the UI for the web application. This included building the chat interface, implementing the logic to connect the LLM to the knowledge base in the vectorstore, and ensuring chat messages were populated correctly. I also designed and implemented the layout, and crucially, connected the LLM to an additional context file uploader. This uploader routes uploaded files to the vectorstore, enabling the LLM to include new context in its responses. Ensured seamless integration between user actions, backend processing, and model inference. The code for these features is included in the project repository and has been thoroughly tested for robustness and usability.
### Bud: 
I am excited to observe real users interact with the system and to collect feedback on the UI/UX. Their input will guide final refinements and help identify any usability bottlenecks.
### Thorn: 
Integrating file uploads with real-time vectorstore updates and ensuring immediate availability for the LLM posed technical challenges, particularly with session state management and error handling in Streamlit.
### Additional thoughts:
Future improvements could include more granular user feedback mechanisms within the UI and web search functionality as well as code writing use cases, where the code and results can be seen within the app.

### Date: 
05/07/2025
### Number of hours: 
3.5
### Rose:
I presented the project at the iShowcase, answered queries from attendees, and demonstrated the full capabilities of the tool. It was rewarding to showcase the impact of our work and receive positive feedback from both peers and faculty. 
### Bud: 
I look forward to exploring opportunities for further development based on showcase feedback, including potential collaborations or extensions of the system for broader research domains.
### Thorn: 
Fielding unexpected questions during the demo highlighted a few edge cases in the application that could be improved, such as handling extremely large document uploads and clarifying citation outputs.
### Additional thoughts:
NA
------------------------------------------------------------------------------------------------


# Weekly/Bi-Weekly Log (Aslam Sheik Dawood)

### Date:  04-04-2025
### Number of hours: 3 
### Rose:
Successfully integrated HuggingFaceEmbeddings and vector store using FAISS. Setting up the foundational embedding model and vector storage was satisfying as it laid the groundwork for the rest of the system.
### Bud: 
Looking forward to implementing query mechanisms over the stored embeddings. 
### Thorn: 
Setting up the right paths and ensuring langchain compatibility with faiss required debugging and reviewing multiple dependencies. 
### Additional thoughts:
Review more use cases for using HuggingFace models within LangChain, particularly for fine-tuned responses.

### Date:  11-04-2025
### Number of hours: 3.5
### Rose:
Worked on document chunking using RecursiveCharacterTextSplitter and successfully created and persisted a FAISS index. This enabled more efficient vector-based retrieval.
### Bud: 
Next goal is to connect a retriever chain with custom prompts for better user interaction. 
### Thorn: 
Splitting strategies had to be adjusted a few times to ensure context retention within chunks. 
### Additional thoughts:
Consider experimenting with chunk overlap for semantic consistency in future iterations.

### Date:  18-04-2025
### Number of hours: 4
### Rose:
Integrated the custom retriever into a conversational retrieval chain. Created custom prompts and tested them with a simple question-answer loop.
### Bud: 
Looking forward to refining prompt templates and deploying the system for broader document sets. 
### Thorn: 
Some prompts were too vague and led to incoherent completions; prompt engineering remains a nuanced task. 
### Additional thoughts:
Try OpenAI's model and compare the performance with HuggingFace in the same chain.

------------------------------------------------------------------------------------------------


# Weekly/Bi-Weekly Log (Pranshu Singh Rawat)

### Date: 
Week number, today's date, etc. 
### Number of hours: 
A quantity of hours, maybe towards specific tasks. 
### Rose:
The highlight from the previous weekly/bi-weekly working period, such as something you found particularly rewarding. This could also be something you're excited to implement now.
### Bud: 
Something that you are looking forward to digging into deeper. This could also be ideas on how to apply concepts to your research in the future. 
### Thorn: 
Something that was challenging that could be worked on, such as anything that wasn't 100% clear and could be elaborated on. Any sticking points should be addressed here. 
### Additional thoughts:
Write anything that you think would be important for YOU later on.

------------------------------------------------------------------------------------------------


# Weekly/Bi-Weekly Log (Vamsi Vadala)

### Date: 
April 8-21, 2025
### Number of hours: 
~20–24 hours

- Embedding setup + storage: ~8 hrs

- Model experimentation (HuggingFace variants): ~6 hrs

- Debugging similarity scores: ~6 hrs

- Documentation and testing: ~4 hrs
### Rose:
Successfully implemented a complete local embedding pipeline using LangChain + HuggingFace + FAISS. I was able to split and store documents locally and validate search functionality with custom text queries. It’s exciting to have a reusable and scalable vector store ready for bio-medical texts.
### Bud: 
I’m looking forward to exploring how I can enhance semantic search accuracy by trying out domain-specific embedding models like BioLinkBERT and PubMedBERT for better results in biomedical research queries.
### Thorn: 
Cosine similarity scores across different models are consistently low (ranging between 0.3–0.7), and results aren’t very sharp even with relevant queries. This could be due to the nature of the text (e.g. BioMedical related terminology, math-heavy context). Understanding how to embed such data more effectively and aligning it with the model's expected input has been a challenge.
### Additional thoughts:
- I want to add evaluation metrics (e.g., manual relevance scores) to compare different models more objectively.

- Would be helpful to test the current FAISS setup against a more interactive retrieval system like RAG or Haystack to assess real-world performance.

- Need to add logging to track document IDs and their matched query results to debug deeper.
