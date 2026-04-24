# Data Scientist technical test

---

## Introduction

The purpose of this technical test is to evaluate a candidate’s ability to process and analyze data, work with an API, and integrate data science techniques. You will be tasked with building a simple RAG (Retrieval-Augmented Generation) solution using OpenAI API hosted on Azure, and your focus will be on the processing of data, optimizing the retrieval quality, evaluating results, and improving the solution’s performance.

### Test duration:
- **Introduction**: 5 minutes to cover test setup and objectives
- **Coding task**: 2 hours of hands-on engineering work

---
## Libraries
Feel free to use libraries you are comfortable with (e.g., **requests** for API calls, **pandas**, **numpy**, **faiss** for similarity search, etc.).


## The Case

### Exercise overview
You are tasked with building an API serving **RAG (Retrieval-Augmented Generation)** technique using the **OpenAI API hosted on Azure**. The solution should interact with a set of documents (e.g., Wikipedia articles, or data you download from a website). It will retrieve relevant information based on user queries and then use the OpenAI model to generate a coherent, informative response based on the retrieved data.

Your RAG API will:
- Accept a **user query** (e.g., a question related to a document)
- Retrieve **relevant data** from the document set
- Use the **OpenAI model** to augment the retrieved data and generate a relevant response

This API (hosted locally) should be ready for integration with a frontend application.

### Key focus areas:
- **Data retrieval**: Efficiently searching and retrieving relevant information from a collection of documents.
- **Data augmentation**: Using the OpenAI API to generate relevant responses based on the retrieved data.
- **Output evaluation**: Evaluating both the retrieval process and the generated response quality.

---

## Tips

Here are some guidelines to help you approach the task:

- **Data preprocessing**: Think about how to preprocess and structure the data to improve retrieval quality. You may want to consider techniques such as cleaning, tokenization, and embeddings.
- **Optimization**: Focus on optimizing both the retrieval and generation phases. How can you make the retrieval more accurate or the response generation more relevant and efficient?
- **Evaluation metrics**: Implement metrics to assess the quality of both the retrieval and generation phases. This could include precision, recall, or any custom evaluation you find useful.
- **API interaction**: Handle API responses efficiently. Make sure to implement error handling for timeouts or incomplete responses.
- **Tool usage**: Use all available tools and libraries to enhance your solution. You are encouraged to research and leverage resources like **ChatGPT** to assist in your development process.

---

## Technical Details

**LLM endpoint on Azure**: Candidates will use an LLM hosted on Azure to provide specific functionalities within the application. The interaction with this LLM should be evident in the backend code.

- **Endpoint for LLM hosted on Azure**: 
  - **GPT-4o-mini**: [https://open-ai-resource-rob.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-08-01-preview](https://open-ai-resource-rob.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-08-01-preview)
  - **Text-embedding**: [https://open-ai-resource-rob.openai.azure.com/openai/deployments/text-embedding-3-large/embeddings?api-version=2023-05-15](https://open-ai-resource-rob.openai.azure.com/openai/deployments/text-embedding-3-large/embeddings?api-version=2023-05-15)
  
- **API Key**: You can find the API key in the **Key Vault storage secrets** in the Azure account that was set up. The KeyVault name is `open-ai-keys-rob`, and the secret name is `open-ai-key-rob`.
  ![image](Azure%20highlighted.png)

- **Azure Account Information**:
  - **Email**: Robapplicant@outlook.com
  - **Password**: TechnicalTester!

- **GitHub**: Make sure to upload your code to a GitHub repository (feel free to use the current GitHub account and create a new repo, or create it in your own GitHub account and add `robonboarding@outlook.com` as a contributor).

If you encounter issues such as permission problems or missing access, let us know as soon as possible, and we'll grant you access to the necessary resources.

---

### Example Python Script for Azure OpenAI Integration

Here is a complete Python script to interact with Azure OpenAI using the **GPT-4o-mini** model for generating a creative tagline:

```python
import os
from openai import AzureOpenAI

class OpenAIChatAssistant:
    """
    A simple class to interact with Azure OpenAI API for generating chat completions.
    """
    def __init__(self, api_key: str, endpoint: str, deployment_name: str = 'gpt-4o-mini'):
        """
        Initialize with the Azure OpenAI credentials.

        Args:
            api_key (str): Azure OpenAI API key.
            endpoint (str): Azure OpenAI endpoint URL.
            deployment_name (str): The deployment name for the OpenAI model.
        """
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-08-01-preview",
            azure_endpoint=endpoint
        )
        self.deployment_name = deployment_name

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response from the model based on the given prompt.

        Args:
            prompt (str): The user query to generate a response.

        Returns:
            str: The AI-generated response.
        """
        # Send the completion request
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
        )
        return response.choices[0].message.content

# Example usage:
if __name__ == "__main__":
    # Replace these values with your actual credentials
    api_key = "YOUR_AZURE_OPENAI_API_KEY"
    endpoint = "YOUR_AZURE_OPENAI_ENDPOINT"

    assistant = OpenAIChatAssistant(api_key=api_key, endpoint=endpoint)
    prompt = "Write a tagline for an ice cream shop."
    
    # Get the response from the model
    response = assistant.generate_response(prompt)
    print("Generated Response: ", response)
```

## Evaluation

### Expectations of the demo

During the demo, you will walk through your solution and demonstrate the following:

- **Functionality**: Ensure that RAG successfully retrieves relevant documents and generates contextually relevant responses using the OpenAI API.
- **Data evaluation**: Share how you’ve evaluated the quality of the retrieval and generation processes, including any metrics you’ve implemented.
- **Optimizations**: Discuss any optimizations you’ve made to improve retrieval accuracy or the generation phase’s performance.
- **Code quality**: Present your code structure and explain your design choices. How would you scale or maintain your solution?

### Evaluation criteria
Your submission will be evaluated based on:

- **Creativity**: Innovative approaches to enhancing the retrieval and generation process.
- **Data processing skills**: Application of data processing techniques to parse and evaluate data effectively.
- **Optimization and efficiency**: Focus on improving both retrieval and generation phases for optimal performance.
- **Code quality**: Clarity, structure, and readability of your code.

---

## Key steps in development

1. **Implement the API**: Create an API that can accept a user query, interact with the retrieval process, and return generated responses based on the OpenAI API. The API should be designed for easy integration with a frontend application.
2. **Prepare the dataset**: Load and preprocess a collection of documents. You can download any text document online (wikipedia article, few chapters of your favourite book, a scientific paper)
3. **Data retrieval**: Implement a process to efficiently search and retrieve relevant documents based on user queries. 
4. **Data augmentation**: Use the OpenAI API to generate responses that are augmented with relevant information from the retrieved documents.
5. **Evaluation**: Develop evaluation metrics to assess the quality of retrieval and response generation, such as precision, recall, or other relevant metrics.
6. **Optimization**: Focus on optimizing retrieval and generation for performance, accuracy, and efficiency.

---

### Getting started
 
- Make sure to check out your resources in azure with the account provided in the technical details
- Start with a very basic poc before adding creative extra's
