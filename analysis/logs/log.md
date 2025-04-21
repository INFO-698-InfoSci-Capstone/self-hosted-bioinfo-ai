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
### Additional thoughts: work with team to stay on track and reach milestones.

------------------------------------------------------------------------------------------------


# Weekly/Bi-Weekly Log (Visalakshi Prakash Iyer)

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
