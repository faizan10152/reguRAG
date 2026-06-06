# Golden Questions Review

Purpose: review these candidate questions for business realism, structural difficulty, and relevance to AI governance work in German business contexts.

Status: approved on 2026-05-31. The canonical machine-readable set is now `data/eval/golden_questions_v1.jsonl`.

This file remains as the human review artifact. The canonical JSONL keeps the same 38 questions and marks each row with `review_status: approved`.

During review, each question could have been marked with one of:

- `keep`
- `edit`
- `drop`

The full set was accepted without per-row edits.

## Evaluation Mix

Target first serious set:

- 38 candidate questions
- HR, finance, healthcare, business operations, legal basics
- English + German
- factual, scenario, multi-hop, cross-lingual, refusal
- structural difficulty labels: exact lookup, semantic paraphrase, multi-source, negative factual, scope ambiguity, source tension, temporal/versioning, out-of-corpus

## Candidate Questions

| ID | Domain | Type | Difficulty | Language | Question | Expected Sources | Review |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GQ001 | HR | scenario | medium | en | A German startup wants to use an AI tool to rank CVs and shortlist candidates. Is this likely to trigger high-risk AI obligations under the EU AI Act? | `eu_ai_act_en`, `eu_ai_act_de` | |
| GQ002 | HR | scenario | medium | en | Our HR team wants to use AI to monitor employee performance and recommend promotions. Which AI Act risk category or obligations should we check first? | `eu_ai_act_en`, `gdpr_en` | |
| GQ003 | HR | scenario | hard | en | If an AI interview tool automatically rejects candidates without human review, which GDPR and AI Act issues should a German company consider? | `gdpr_en`, `eu_ai_act_en`, `bdsg_de` | |
| GQ004 | HR | factual | medium | de | Welche Datenschutzregeln sind relevant, wenn ein deutsches Unternehmen Beschäftigtendaten für ein KI-System verarbeitet? | `gdpr_de`, `bdsg_de` | |
| GQ005 | HR | scenario | medium | en | What should a company document before deploying an AI system that evaluates employees in a work-related context? | `eu_ai_act_en`, `gdpr_en` | |
| GQ006 | Finance | scenario | medium | en | A fintech wants to use AI to assess creditworthiness for consumer loans. Could this be considered high-risk under the AI Act? | `eu_ai_act_en` | |
| GQ007 | Finance | scenario | hard | en | A bank uses transaction data to train a fraud-detection model. Which GDPR principles should guide the data use? | `gdpr_en`, `dsk_ki_datenschutz_de` | |
| GQ008 | Finance | multi-hop | hard | en | If a loan application is rejected solely by an automated model, what rights or safeguards should the customer be told about? | `gdpr_en`, `eu_ai_act_en` | |
| GQ009 | Finance | factual | medium | en | Which GDPR article is most relevant for the lawfulness of processing personal data in an AI risk-scoring workflow? | `gdpr_en` | |
| GQ010 | Finance | scenario | medium | de | Ein Versicherer nutzt KI zur Risikobewertung von Kundinnen und Kunden. Welche Datenschutz- und Transparenzfragen sollte das Unternehmen prüfen? | `gdpr_de`, `eu_ai_act_de` | |
| GQ011 | Healthcare | scenario | medium | en | A hospital wants to use AI to prioritize patient triage requests. Could this fall into a high-risk category? | `eu_ai_act_en`, `gdpr_en` | |
| GQ012 | Healthcare | scenario | hard | en | Can a clinic train an AI assistant on patient notes that contain health data? Which GDPR concerns are most important? | `gdpr_en`, `dsk_ki_datenschutz_de` | |
| GQ013 | Healthcare | factual | medium | en | Which GDPR rule is relevant when processing health data or other special categories of personal data? | `gdpr_en` | |
| GQ014 | Healthcare | scenario | hard | de | Eine Klinik möchte Patientenanfragen mit einem LLM zusammenfassen. Welche Risiken entstehen bei personenbezogenen Gesundheitsdaten? | `gdpr_de`, `dsk_ki_datenschutz_de` | |
| GQ015 | Healthcare | multi-hop | hard | en | If an AI system helps dispatch emergency response resources, what AI Act risk concerns should be checked? | `eu_ai_act_en` | |
| GQ016 | Business Ops | scenario | medium | en | A company wants to train an internal chatbot using customer support tickets. What GDPR questions should be answered before training? | `gdpr_en`, `bfdi_ki_fragen_de`, `dsk_ki_datenschutz_de` | |
| GQ017 | Business Ops | factual | easy | en | What are the GDPR principles of purpose limitation and data minimisation, and why do they matter for AI systems? | `gdpr_en` | |
| GQ018 | Business Ops | scenario | medium | en | Employees use a third-party AI writing assistant with internal company data. What should the company check before rollout? | `gdpr_en`, `bnetza_ai_literacy_de`, `eu_ai_act_en` | |
| GQ019 | Business Ops | scenario | medium | en | A marketing team creates AI-generated images that look like real people. Are there transparency obligations to consider? | `eu_ai_act_en`, `bnetza_ai_en` | |
| GQ020 | Business Ops | scenario | hard | de | Ein Unternehmen möchte einen internen KI-Assistenten für Verträge und Kundendaten einsetzen. Welche Datenschutz- und KI-Kompetenzfragen sind relevant? | `gdpr_de`, `bnetza_ai_literacy_de`, `eu_ai_act_de` | |
| GQ021 | Legal Basics | factual | easy | en | What is AI literacy under the EU AI Act, and who should have it? | `eu_ai_act_en`, `bnetza_ai_literacy_de` | |
| GQ022 | Legal Basics | factual | easy | de | Was bedeutet KI-Kompetenz nach der KI-Verordnung? | `bnetza_ai_literacy_de`, `eu_ai_act_de` | |
| GQ023 | Legal Basics | factual | medium | en | When is a data protection impact assessment required under GDPR? | `gdpr_en`, `gdpr_de` | |
| GQ024 | Legal Basics | factual | medium | en | What is the difference between a provider and a deployer under the AI Act? | `eu_ai_act_en` | |
| GQ025 | Legal Basics | factual | medium | en | What does the system need to know before answering whether an AI system is high-risk? | `eu_ai_act_en`, `bnetza_ai_en` | |
| GQ026 | Cross-lingual | factual | medium | en | What does German regulator guidance say about building AI literacy inside an organization? | `bnetza_ai_literacy_de` | |
| GQ027 | Cross-lingual | scenario | hard | de | Dürfen Kundensupport-Tickets mit personenbezogenen Daten für das Training eines KI-Modells verwendet werden? | `gdpr_de`, `bfdi_ki_fragen_de`, `dsk_ki_datenschutz_de` | |
| GQ028 | Cross-lingual | scenario | hard | en | A German company uses an English-language vendor tool for employee analytics. Which German or EU sources should it consult first? | `eu_ai_act_en`, `gdpr_en`, `bdsg_de` | |
| GQ029 | Refusal | unanswerable | easy | en | What is the exact penalty under Article 999 of the EU AI Act? | none; should refuse | |
| GQ030 | Refusal | unanswerable | medium | en | Does BaFin guidance in this corpus allow our credit-scoring model to go live next week? | none; should refuse because BaFin guidance is not in corpus | |
| GQ031 | Refusal | unanswerable | medium | en | Give binding legal advice on whether our hospital AI deployment is fully compliant. | should refuse / qualify as non-legal advice | |
| GQ032 | Refusal | unanswerable | hard | de | Welche Entscheidung hat ein deutsches Gericht zu unserem konkreten KI-Bewerbungstool getroffen? | none; should refuse | |
| GQ033 | Legal Basics | source tension | hard | en | GDPR Article 22 and the AI Act both address automated decision-making. Do they impose the same obligations, or are there differences a company must track separately? | `gdpr_en`, `eu_ai_act_en` | |
| GQ034 | HR | scope ambiguity | hard | en | A SaaS vendor provides an AI scoring component embedded in our HR software. Who holds the provider obligations under the AI Act: the vendor or us? | `eu_ai_act_en` | |
| GQ035 | Legal Basics | temporal/versioning | hard | en | Did the AI Act obligations for high-risk systems change between the initial proposal and the final adopted version? | should refuse or qualify because corpus contains adopted sources, not legislative history | |
| GQ036 | Legal Basics | negative factual | medium | en | Does the GDPR require a DPIA for every AI system that processes personal data? | `gdpr_en`; expected answer should say no, only where processing is likely high-risk | |
| GQ037 | Business Ops | process/operational | medium | en | What technical documentation must a provider of a high-risk AI system maintain under the AI Act? | `eu_ai_act_en` | |
| GQ038 | Refusal | out-of-corpus | medium | en | What does the German Works Council Act say about works council consent for AI deployment? | none; should refuse because Betriebsverfassungsgesetz is not in corpus | |

## Questions To Answer During Review

1. Which questions feel most realistic for the roles you want?
2. Which domains should be emphasized more: HR, finance, healthcare, or business ops?
3. Are the German questions natural enough, or should we rewrite them?
4. Are any questions too legalistic and not AI-engineering focused?
5. Which questions would you feel comfortable explaining in an interview?
6. Which questions are hard for different structural reasons, not just because they mention more sources?
7. Should we trim one finance factual question such as GQ009 after adding the stronger structural cases?

## Next Step After Review

For approved questions, the benchmark layer now includes exact expected citation labels for the first 12 high-value questions. Future iterations will add:

- reference answers
- required claims
- forbidden unsupported claims
- retriever difficulty labels
