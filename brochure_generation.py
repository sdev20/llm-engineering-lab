import json
from config import CHUNK_OVERLAP_TOKENS_SIZE, MAX_LINKS_PER_PAGE, API_MODEL, CHUNK_TOKENS_SIZE, RAG_ENABLED
from prompts import get_brochure_system_prompt, get_brochure_user_prompt, get_relevant_links_system_prompt, get_relevant_links_user_prompt
from llm_client import client
from rag import chunk_pages, embed_texts
from scraper import fetch_website_content, fetch_website_links



model = API_MODEL


def fetch_relevant_links(links, website_url):
    """
    Fetches relevant links for brochure generation from the given list of links.

    Args:
        links (list): The list of links to process.
        website_url (str): The URL of the website to scrape.
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": get_relevant_links_system_prompt()},
            {"role": "user", "content": get_relevant_links_user_prompt(links, website_url)}
        ],
        response_format={"type": "json_object"} # Specify the response format as JSON
    )

    relevant_links = response.choices[0].message.content
    return json.loads(relevant_links)  # Parse the JSON string into a Python dictionary

def fetch_content_and_relevant_links(website_url):
    """
    Fetches the content of the given website URL and relevant links for brochure generation.

    Args:
        website_url (str): The URL of the website to scrape.

    Returns:
        list: A list of pages with their content and URLs.
    """

    # fetch the list of links from the website
    website_links = fetch_website_links(website_url) 
    # fetch the relevant links for brochure generation
    relevant_links = fetch_relevant_links(website_links, website_url)

    pages = [{"type": "landing", "content": fetch_website_content(website_url), "url": website_url}]

    for link in relevant_links['links'][:MAX_LINKS_PER_PAGE]:
       pages.append({"type": link['type'], "content": fetch_website_content(link["url"]), "url": link["url"]})
    return pages


def create_brochure(website_url):
    """
    Creates a brochure for the given website URL.

    Args:
        website_url (str): The URL of the website to scrape.
    """
    pages = fetch_content_and_relevant_links(website_url)
    # if RAG_ENABLED:
    #     print("RAG Enabled")
    #     chunks = chunk_pages(pages, CHUNK_TOKENS_SIZE, CHUNK_OVERLAP_TOKENS_SIZE)
    #     chunks_embeddings = embed_texts([c['content'] for c in chunks])
    #     print(chunks_embeddings)
    #     print(chunks_embeddings.shape)
    company_content = format_pages_for_brochure(pages)

    system_prompt = get_brochure_system_prompt() 
    user_prompt = get_brochure_user_prompt(company_content)

    brochure_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )

    result = brochure_response.choices[0].message.content
    return result
    #return ""


def stream_brochure(website_url):
    """
    Creates a brochure for the given website URL.

    Args:
        website_url (str): The URL of the website to scrape.
    """
    pages = fetch_content_and_relevant_links(website_url)
    company_content = format_pages_for_brochure(pages)
    brochure_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": get_brochure_system_prompt()},
                    {"role": "user", "content": get_brochure_user_prompt(company_content)}
                ],
                stream=True
            )

    result = ""
    for chunk in brochure_response:
        if chunk.choices[0].delta.content is not None:
            delta = chunk.choices[0].delta.content
            result += delta
            print(delta, end="", flush=True)

    return result


def stream_brochure_for_ui_chat(website_url):
    """
    Creates a brochure for the given website URL.

    Args:
        website_url (str): The URL of the website to scrape.
    """
    pages = fetch_content_and_relevant_links(website_url)
    company_content = format_pages_for_brochure(pages)
    brochure_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": get_brochure_system_prompt()},
                    {"role": "user", "content": get_brochure_user_prompt(company_content)}
                ],
                stream=True
            )

    result = ""
    for chunk in brochure_response:
        if chunk.choices[0].delta.content is not None:
            delta = chunk.choices[0].delta.content
            result += delta
            yield result # if called from gradio, this will stream the result to the UI in real-time
            

def format_pages_for_brochure(pages):
    formatted_content = f"## Landing Page:\n\n{pages[0]['content']}\n## Relevant Links:\n"
    for page in pages[1:]:
        formatted_content += f"\n\n### Link: {page['type']}\n"
        formatted_content += page['content']
    return formatted_content


