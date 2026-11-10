# LLM Engineering Lab

A hands-on lab where I learn LLM engineering by building and measuring real things, and write down why each decision was made. The first project is a **sales brochure generator**: it reads a company's public website and writes a short Markdown brochure for prospective customers, investors, and recruits. Further experiments (RAG, tool calling, UIs) are built on top of it.

## What this project demonstrates

| Skill | Where to look |
|---|---|
| Calling LLM APIs, including streaming responses | `brochure_generation.py`, `llm_client.py` |
| Structured output: asking the model for JSON (link selection) | `brochure_generation.py` |
| Token budgeting and cost control, backed by measurements on real sites | `tokens.py`, `config.py`, [Design decisions](#design-decisions) |
| RAG building blocks: token-based chunking, embeddings, similarity ranking (in progress) | `rag.py`, [Flow being built](#flow-being-built-rag-experiment) |
| Tool calling and an agent-style loop, including assembling streamed tool-call fragments | `SalesBrochureWithUI/conversational_AI_with_ui.py` |
| Gradio UIs: a streaming form and a chat | `SalesBrochureWithUI/` |
| Web scraping with `requests` and BeautifulSoup | `scraper.py` |
| Engineering judgment: trade-offs, known issues and limitations documented openly | [Design decisions](#design-decisions), [Limitations](#limitations) |

The brochure generator is the vehicle; the focus is the engineering decisions around it.

## Design focus: token budgets and cost control

The brochure itself is simple to produce. Most of the engineering effort here goes into deciding **how much website text the model should read**, because that drives cost, latency, and quality. The recurring questions are:

- How many tokens does each page cost, and what is the total per request?
- What do we keep when a site is bigger than the budget?
- How do we choose what to keep so the useful facts survive?

Decisions below are made from measurements on real sites (google.com, cellares.com, huggingface.com), not guesses. In LLM-engineering terms, the limits are *resource guardrails* (cost and size caps). There are no safety or quality guardrails yet (see Limitations).

## How it works

### Flow that runs today

```
URL
 → fetch landing page + collect its links             (scraper.py)
 → LLM picks the relevant links, returns JSON         (brochure_generation.py)
 → fetch each selected page, keep first MAX_PAGE_TOKENS tokens   (scraper.py, tokens.py)
 → join pages into one prompt                          (brochure_generation.py)
 → LLM writes the brochure                             (brochure_generation.py)
```

This is the **baseline**: simple and predictable, but it always keeps the *start* of each page, whether or not that part is useful.

### Flow being built (RAG experiment)

```
URL
 → fetch landing page + selected links, pages kept (nearly) uncut
 → split every page into overlapping token chunks      (rag.py, done)
 → embed every chunk and a fixed brochure query        (planned)
 → rank chunks by cosine similarity to the query       (planned)
 → fill MAX_CONTENT_TOKENS with the best chunks,       (planned)
   restore page and reading order
 → format into the prompt → LLM writes the brochure
```

Progress:

| Step | What | Status |
|---|---|---|
| R1 | Chunk page text in tokens, with overlap, keeping page type, URL and position | Done (`rag.py`) |
| R2 | Embed chunks and the query | In progress (`embed_texts` in `rag.py` works; query embedding and caching next) |
| R3 | Cosine similarity ranking | Planned |
| R4 | Fill the token budget with top chunks, restore order | Planned |
| R5 | Compare against the baseline on the same cached pages | Planned |
| R6 | Tune chunk size, overlap, query wording, diversity, near-duplicates | Planned |

## Project layout

| File | Purpose |
|---|---|
| `main.py` | Command-line entry point |
| `brochure_generation.py` | Pipeline: link selection, page fetching, brochure generation (normal and streaming) |
| `scraper.py` | Fetches links and page text with `requests` and BeautifulSoup |
| `tokens.py` | Token counting, truncation, and encode/decode helpers (`tiktoken`) |
| `rag.py` | Token-based chunking (`chunk_text`, `chunk_pages`) and embeddings (`embed_texts`); ranking will go here |
| `llm_client.py` | The single shared OpenAI client used by the pipeline, the RAG code and the UI |
| `prompts.py` | System and user prompts |
| `config.py` | Model name and size limits |
| `SalesBrochureWithUI/stream_brochure_with_ui.py` | Gradio form: enter a URL, watch the brochure stream in |
| `SalesBrochureWithUI/conversational_AI_with_ui.py` | Gradio chat that decides by itself when to generate a brochure (tool calling) |

## Requirements

- Python 3
- An OpenAI API key with access to the configured model
- Network access to the target website and the OpenAI API

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create a `.env` file in the project directory with your API key:

```dotenv
OPENAI_API_KEY=your-openai-api-key
```

To try the RAG path (see below), add this optional line to `.env`:

```dotenv
RAG_ENABLED=true
```

The model name and size limits are set in `config.py`, not in `.env`.

## Generate a brochure

```bash
python main.py https://example.com
```

Add `--stream` to print the brochure as it is generated. Without it, the full brochure is printed at the end. The brochure is not saved to a file.

## Web UI (Gradio)

Two small Gradio apps live in `SalesBrochureWithUI/`. Run them **from the project root**, because they import `config`, `llm_client` and `brochure_generation` from the root folder.

Plain URL form (streams the brochure):

```bash
python -m SalesBrochureWithUI.stream_brochure_with_ui
```

Chat that calls the brochure generator as a tool when the user asks for one:

```bash
python -m SalesBrochureWithUI.conversational_AI_with_ui
```

To reload automatically when you save a file, use the Gradio CLI. It runs the file as a script, so add the project root to `PYTHONPATH`, and tell it the name of the app object (`view`):

```bash
PYTHONPATH=. gradio SalesBrochureWithUI/conversational_AI_with_ui.py --demo-name view
```

Why `-m` and `PYTHONPATH`: running `python path/to/file.py` only puts that file's own folder on `sys.path`, so `from config import ...` fails with `ModuleNotFoundError`. `python -m Package.module` from the root, or `PYTHONPATH=.`, makes the root-level modules importable.

How the chat works (`conversational_AI_with_ui.py`):

1. The user message goes to the model together with a tool schema describing `stream_brochure_for_ui_chat(website_url)`.
2. The reply is streamed. Plain text is shown as it arrives; a tool call arrives in fragments (`id` and name in the first, JSON `arguments` in pieces), so fragments are merged by `index` and the arguments are parsed only after the stream ends.
3. The function is looked up by name in `AVAILABLE_FUNCTIONS` and called. A generator tool (`inspect.isgeneratorfunction`) is iterated and each partial text is shown live; a normal tool's result is shown as one string.
4. Gradio replaces the message on every yield, so the chat keeps a running string and shows earlier tools' finished text above the one that is streaming.

## Configuration (`config.py`)

| Setting | Value | Meaning |
|---|---|---|
| `API_MODEL` | `gpt-4o-mini` | Model used for link selection and brochure generation; also selects the `tiktoken` encoding |
| `MAX_LINKS_PER_PAGE` | 10 | Maximum linked pages fetched in addition to the landing page |
| `MAX_PAGE_TOKENS` | 2,500 | Tokens kept per page in the baseline |
| `MAX_CONTENT_TOKENS` | 25,000 | Limit for all scraped content in one request (to be enforced by the RAG budget step) |
| `CHUNK_TOKENS_SIZE` | 350 | Tokens per chunk |
| `CHUNK_OVERLAP_TOKENS_SIZE` | 50 | Tokens shared between neighbouring chunks |
| `RAG_ENABLED` | `false` | Read from `.env`. When true, pages are fetched without the per-page cap and chunked; when false, the baseline runs |

## Design decisions

### Limits and tokens

**Why limit scraped content at all?** The context window is large (200k tokens for the model I checked), so the limits are not about fitting. They keep cost and latency down, stop noisy text from diluting the useful facts, and protect against unusually large pages or link lists.

**Why a token limit and not a character limit?** Tokens per character vary a lot. On huggingface.com the pricing and enterprise pages hit the old 8,000-character cap but cost 7,240 and 6,362 tokens (about 1.1 characters per token), while ordinary pages cost about 1,500 tokens (about 4 characters per token). A character cap cannot predict cost, so a token cap, measured with `tiktoken`, matches what the model reads and what is billed.

**Why 2,500 tokens per page (baseline)?** Most pages measured 250 to 2,000 tokens, so 2,500 leaves them whole and only trims outliers. It is also close to an even share of the budget: 25,000 tokens across 11 pages is about 2,270.

**Why 25,000 tokens in total?** Measured inputs ranged from about 5k to 22k tokens. 25k covers that range with some room. It is a cost and quality guard, not a context-window limit. Output is not a concern: briefs were 484 to 685 tokens whatever the input size.

**Why was the old 5,000-character prompt cut removed?** It came from a tutorial example. It cut the combined text, so it silently dropped the linked pages, which are the content the link-selection step exists to collect. Prompt-building code now only builds the prompt; limits are applied to the scraped content before it.

### Why RAG

The baseline's per-page cut keeps the first 2,500 tokens of every page, so what survives depends on position, not relevance. Uncut, the real page sizes turned out to be larger than the old caps showed:

- Hugging Face pricing is about 15k tokens and enterprise about 12k, versus 7,240 and 6,362 when capped.
- Those two pages are 93 of 130 chunks (about 70%). All pages together come to about 43.7k chunk tokens, against a 25k budget.

So about 40% of the content has to go, and the question is which 40%. Ranking chunks by similarity to a brochure-focused query chooses by relevance instead of by position or page length, and it can pick useful passages from anywhere in a long page. This is also how unpredictable page types are ranked: the link-selection model invents the type names, so ranking by type would not work across sites.

The baseline is kept so the two approaches can be compared on the same pages.

### Chunking

**Why chunk in tokens and not characters?** For the same reason as the page limits: a chunk of 1,200 characters can be 300 or 1,000 tokens. Token-sized chunks are evenly sized, fit the embedding model's input limit, and make the budget arithmetic exact. Token boundaries decide the cut; each slice is decoded back to text, because embeddings and prompts take text.

**Why chunk at all?** Embedding a whole page gives one vector that blends every topic on it, so ranking could only choose whole pages. Chunks let retrieval pick the right passages.

**Why 350 tokens with 50 overlap?** Starting values: large enough to hold a complete idea, small enough to stay on one topic. Overlap keeps a sentence that falls on a boundary whole in a neighbouring chunk. It costs about 14% more tokens to embed (130 chunks hold 43.7k tokens, but about 37.7k are unique). If two adjacent chunks are both selected, the overlapping text appears twice in the prompt, which the budget step has to handle. Both numbers are to be tuned in R6.

**Why does every chunk carry `url`, `type`, `content`, and `index`?** After ranking, chunks from different pages are mixed. The page type is needed for prompt headings, `index` restores reading order within a page, and the source page is needed to judge which pages the winning chunks came from. `index` restarts at 0 for each page.

**Why no vector database?** A page set is 100 to 250 chunks and is rebuilt for each run, so a Python list and `numpy` are enough, and they keep the mechanics visible.

**Why must pages reach `rag.py` (nearly) uncut?** If the scraper has already cut a page to 2,500 tokens, retrieval can never choose the later parts. The baseline cap therefore must not apply to the RAG path; only a large safety cap should, so one huge page cannot flood the pipeline.

### Chat UI and tool calling

**Why tool calling and not URL detection?** A regex on the message would treat any pasted link as "make a brochure" ("what does https://x.com do?") and would miss phrasing like "do one for stripe's site". With a tool schema the model decides whether the user wants a brochure and extracts the URL itself.

**Why stop after the tool and not send the result back to the model?** The tool already runs its own LLM call and its output is the final answer. Feeding thousands of tokens of finished brochure back for a second model call adds cost and latency, and the model might paraphrase it. The full round trip (assistant `tool_calls` message, then a `tool` message with the matching `tool_call_id`, then another call) is only worth it for tools that return raw data the model must explain.

**Why does the model pick a tool by name and description only?** It never sees the Python functions, so it cannot know whether one streams. Streaming is a UI concern: expose one tool per purpose and point its name at the implementation you want.

**Why does `chat` yield instead of return?** `chat` is a generator, so a `return value` ends it and the value is discarded; Gradio then shows an empty reply with no error. Results must be `yield`ed (or forwarded with `yield from`).

## Measurements

Baseline input and output (before per-page token caps):

| Site | Input tokens | Output tokens |
|---|---|---|
| google.com | 4,706 | 558 |
| cellares.com | about 15,100 | 685 |
| huggingface.com | 22,028 | 484 |

Chunking huggingface.com with the uncut scraper: 11 pages, 130 chunks, 43,664 chunk tokens. Pricing (51 chunks) and enterprise (42 chunks) dominate.

## Work in progress and known issues in the code

- With `RAG_ENABLED=true`, pages are fetched **without** any cap and chunked, but the chunks are not used yet: all page text is still sent to the model. Nothing is ranked or limited until R2 to R4 are connected, so a run with the flag on can exceed `MAX_CONTENT_TOKENS`. With the flag off (the default), the capped baseline runs for both normal and streaming modes.
- There is no RAG safety cap in `config.py` yet.
- The chat UI has no tool-call limit: if the model asks for several tools in one response they run one after another. `parallel_tool_calls=False` on the request would restrict it to one.
- Brochure generation in the chat can sit empty for a while, because scraping and link selection happen before the first streamed text.
- `stream_brochure_with_ui.py` still has leftover tutorial code (`message_gpt`, `stream_gpt`, the example row and the "Message Shouter" title).
- The link-selection step is not deterministic, so two runs on the same site can fetch different pages. Cache fetched pages (for example to a JSON file) when comparing baseline and RAG.
- The last chunk of a page can be tiny, with only a token or two of new text.

## Limitations

- The scraper reads HTML and does not run JavaScript, so script-rendered sites return little text (google.com's landing page is about 33 tokens; a Workable careers page gave one chunk). With thin input the model may fill gaps from what it already knows, so check the brochure against the source.
- Only links on the landing page are followed, links are not normalised or deduplicated, and the model sometimes selects near-duplicates (for example several news articles on cellares.com).
- Navigation, footer, and similar boilerplate text is not removed from pages.
- Website requests can fail because of network errors or site restrictions, and there is no request timeout yet.
- Guardrails: only resource limits (token caps) exist. There is no protection against prompt injection from scraped text, no check that the brochure sticks to the source, no validation of the link JSON, and no block on private or local URLs.
