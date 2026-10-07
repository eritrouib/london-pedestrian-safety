# Switching on the AI assistant

The dashboard's **Ask the data** box needs a small relay that holds your Anthropic API key. Until it's set up, visitors see: *"The AI assistant is currently unavailable. You can still explore everything with the filters and map below."* Everything else keeps working.

About 20 minutes, all in a web browser. No command line except the last step.

## How it works

1. A visitor types a question in the dashboard.
2. The dashboard sends it to your **relay** (a free Cloudflare Worker), which holds your API key so it never appears in the web page.
3. The relay asks **Claude Haiku 4.5** to translate the question into a structured query, for example *borough = Camden, age 0–15, hours 0–6*. Claude does not calculate anything.
4. The dashboard checks the query, computes every number from your data in the visitor's browser, and moves the map to match.

## What it costs

- **Visitors:** nothing.
- **Cloudflare:** free (the free plan allows 100,000 requests a day).
- **Anthropic API:** pay-as-you-go, separate from any Claude subscription. Each question is one short call to the cheapest model, well under 1p. A few pounds of credit covers hundreds to thousands of questions.

## 1. Anthropic: get an API key

1. Go to **console.anthropic.com** and sign up (you can use the same email as your Claude account; billing is separate).
2. **Billing:** add a small amount of credit, e.g. $5. **Leave auto-reload off.** When the credit runs out, the assistant shows the "unavailable" message instead of costing more.
3. Optional: in **Limits**, set a monthly spend limit as an extra safety net.
4. **API keys → Create key.** Name it `london-dashboard`. Copy the key now (it's shown once) and keep it private. Never put it in GitHub.

## 2. Cloudflare: create the relay

1. Go to **dash.cloudflare.com** and sign up (free).
2. **Workers & Pages → Create → Create Worker.** Name it `lps-ai` and click **Deploy** (it deploys a "Hello World" placeholder).
3. Click **Edit code**. Delete everything in the editor, paste in the contents of `ai/worker.js`, and click **Deploy**.
4. Go to the Worker's **Settings → Variables and Secrets** and add:

   | Type | Name | Value |
   |---|---|---|
   | Secret | `ANTHROPIC_API_KEY` | your key from step 1 |
   | Text | `ALLOWED_ORIGIN` | `https://eritrouib.github.io` |
   | Text | `PER_VISITOR_DAILY` | `20` (questions per visitor per day) |

5. Copy the Worker's address from its overview page. It looks like `https://lps-ai.<your-name>.workers.dev`.

### Optional: a total daily cap

The per-visitor limit is approximate on its own. For an exact limit and a total cap per day:

1. **Storage & Databases → KV → Create** a namespace called `lps-usage`.
2. In the Worker: **Settings → Bindings → Add → KV namespace**, variable name `USAGE`, namespace `lps-usage`.
3. Add a Text variable `DAILY_CAP`, e.g. `200`.

## 3. Connect the dashboard

In your project folder, once:

```
python scripts/06_build_dashboard.py --ai-url https://lps-ai.<your-name>.workers.dev
git add .
git commit -m "Switch on the AI assistant"
git push
```

The address is saved in `config/ai_endpoint.txt`, so later rebuilds remember it. After a minute, open the dashboard, press **Ctrl + F5** and try an example question.

## Optional: a "buy me a coffee" link

Visitors can chip in towards the running cost. When set, a short line appears under the Ask box, and in the messages shown when the assistant is out of credit or has hit its limits.

1. Create a page at **buymeacoffee.com** (or a similar service such as Ko-fi). You'll get a link like `https://buymeacoffee.com/yourname`.
2. Once:
   ```
   python scripts/06_build_dashboard.py --support-url https://buymeacoffee.com/yourname
   ```
   It's saved in `config/support_url.txt`. Commit and push as usual.

## Day to day

- **Turn it off instantly:** delete the `ANTHROPIC_API_KEY` secret in Cloudflare. Visitors see the "unavailable" message.
- **See usage and cost:** Anthropic Console → Usage. Cloudflare → your Worker → Metrics / Logs.
- **If answers stop:** usually credit has run out (Console → Billing), or the key was deleted.

## Messages visitors may see

| Situation | Message |
|---|---|
| Not set up, no credit, or turned off | The AI assistant is currently unavailable. You can still explore everything with the filters and map below. |
| Anthropic briefly overloaded | The AI assistant is busy right now. Please try again in a minute… |
| Visitor reached their daily limit | You've reached today's limit of questions… |
| Total daily cap reached | The AI assistant has answered all the questions it can for today… |
| Question the data can't answer | A short explanation from the assistant, e.g. that the data doesn't describe vehicles or drivers |

## Safety by design

- The API key lives only in Cloudflare, never in the web page or on GitHub.
- The relay only answers requests from your dashboard's address, and limits how often each visitor can ask.
- Claude only returns a query. The dashboard checks every field against a fixed list (known boroughs, years, severities, hour and age ranges) and ignores anything else, so a misbehaving or manipulated reply can't inject content or invent numbers.
- Each answer starts with **"I read this as…"** and restates exactly what was counted, so a misread question is easy to spot.
