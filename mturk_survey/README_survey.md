## MTurk Survey Link + Hosted Survey Page

MTurk only stores an external survey link, and the actual survey page is served from own web server.

---

## File structure

```
mturk_survey_link_external_site/
  start_survey_server.py          # starts the web server and saves results
  public/
    survey.html                   # survey page with original/mutated highlighting
    index.html                    # local batch list viewer
    batches/*.json                # 39 HIT batch definitions
  data/
    mturk_survey_links_pilot_TEMPLATE.csv
    mturk_survey_links_all_TEMPLATE.csv
  scripts/
    make_survey_link_csv.py
    merge_server_results_with_mturk.py
  results/server_saved/           # JSON saved automatically after worker submits
```

---

## 1. Test locally first

On macOS / Linux:

```bash
python3 start_survey_server.py --host 127.0.0.1 --port 8000
```

On Windows:

```powershell
py start_survey_server.py --host 127.0.0.1 --port 8000
```

Open the browser:

```
http://localhost:8000/index.html
```

Click a batch, for example:

```
http://localhost:8000/survey.html?batch=sst2_001
```

After submission, if the page shows a short completion code like:

```
SGV-sst2_001-AB12CD34
```

then the server saved the response successfully. Results are stored in:

```
results/server_saved/
```

---

## 2. Make it accessible to remote MTurk workers

The easiest option is Cloudflare Tunnel. Keep the local server running, then open a new terminal:

```bash
cloudflared tunnel --url http://localhost:8000
```

It gives you an HTTPS URL such as:

```
https://xxxx.trycloudflare.com
```

Use that URL as the `base-url` when generating the MTurk CSV.

If you have a public server, upload the folder and run:

```bash
python3 start_survey_server.py --host 0.0.0.0 --port 8000
```

Then use the server’s public HTTPS/HTTP address. HTTPS is recommended for live MTurk.

---

## 3. Generate the Amazon MTurk Survey Link CSV

With your public URL, run:

```bash
python3 scripts/make_survey_link_csv.py --base-url https://xxxx.trycloudflare.com --out data/mturk_survey_links_pilot.csv --pilot
python3 scripts/make_survey_link_csv.py --base-url https://xxxx.trycloudflare.com --out data/mturk_survey_links_all.csv
```

Use the pilot CSV first:

```
data/mturk_survey_links_pilot.csv
```

Then use the full CSV for the main run:

```
data/mturk_survey_links_all.csv
```

The important column in the CSV is:

```
survey_url
```

Each row is the web link for one HIT.

---

## 4. Create the Project in MTurk Requester

Use MTurk’s Survey Link template. This setup is designed for that flow.

When creating the project:

```
Create -> New Project -> Survey Link
```

Set the Survey Link / Survey URL field to:

```
${survey_url}
```

Keep the completion code field so workers can paste the code after finishing the external page.

Publish the batch and upload:

```
data/mturk_survey_links_pilot.csv
```

Workers will see an external link. After they complete the survey page, they paste the completion code back into MTurk.

---

## 5. Verify before running production

- Do not close the server terminal.
- If using Cloudflare Tunnel, do not close that terminal either.
- Keep the computer awake.
- Start with only 3 pilot HITs.
- Confirm JSON files appear in `results/server_saved/`.
- Confirm MTurk’s downloaded CSV contains workers’ pasted `SGV-...` codes.

---

## 6. Merge MTurk results with saved survey data

After MTurk completes, download the results CSV, e.g.:

```
mturk_results.csv
```

Run:

```bash
python3 scripts/merge_server_results_with_mturk.py --mturk-csv mturk_results.csv --results-dir results/server_saved --outdir results/merged
```

This produces:

```
results/merged/assignment_summary.csv
results/merged/item_annotations.csv
```

`item_annotations.csv` contains the item-level data for majority vote and agreement analysis.

---

## 7. Backup if server saving fails

If the server fails, the page shows a long `SGVJSON-...` backup code and automatically downloads JSON. That backup code contains the answers and can be parsed from the MTurk CSV, but it is much better to ensure the server saves successfully.

---

## 8. Important note

This method does not use `batch_b64` and does not require uploading the 39 batch JSON files to MTurk. The JSON remains on your hosted survey server.

MTurk only needs the `survey_url` CSV.