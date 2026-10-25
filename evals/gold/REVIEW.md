# Gold review sheet (3 calibration resumes, parse v7)

Grades: correct / partial / wrong / hallucinated / missing. The **Human** column is the gold in `parsing_human.jsonl`; edit that file, then rerun `python scripts/build_review_sheet.py`. The Judge column is from `parsing-20261025T211457Z.json`; ⚠ marks a disagreement.

## Chen Li  (`e89dc25ec722`)

| Field | Parsed value | Human | Judge | Reason |
|---|---|---|---|---|
| name | Chen Li | **correct** | correct | 'Chen Li' matches the heading; the '(Alex)' nickname is correctly left out. |
| contact | {"email": "Alex_chen2024@gmail.com", "phone": "+852 65478923", "location_text": null} | **correct** | correct | Email Alex_chen2024@gmail.com and phone +852 65478923 match the header; the text gives no street or city, so location_text null is right. |
| education | University of Hong Kong: Master of Science in Financial Technology (Masters); University of Hong Kong: Bachelor of Business Administration (Finance) (Bachelors) | **correct** | correct | Both HKU degrees are present: MSc Financial Technology Sept 2021 to Jul 2023 and BBA Finance Sept 2017 to Jul 2021; dates parse to 2021-09/2023-07 and 2017-09/2021-07, country HK. |
| experience | 6 roles | **correct** | correct | All six roles are present: Bank of China analyst and intern, CUHK research assistant, Meridian Asia Capital intern, Fujia Aid Teaching Society and the swimming club. None is invented. |
| skills | {"programming": ["Python", "R", "SQL", "VBA"], "tools": ["Excel", "PowerPoint", "Word", "Bloomberg Terminal", "WIND Database", "GIS mapping tools"], "domain": [… | **correct** | correct | v7 parse with skill dedup: Python, R, SQL, VBA, Excel, PowerPoint, Word, Bloomberg, WIND and GIS tools are right; the 'Microsoft Office Suite' umbrella is dropped because its members are listed, and course names are gone. |
| certifications | [] | **correct** | correct | The resume has no certification. Dean's List and other awards are correctly not treated as certifications. |
| languages | [{"name": "English", "level": "fluent"}, {"name": "Mandarin", "level": "native"}, {"name": "Cantonese", "level": "fluent"}] | **correct** | correct | English fluent, Mandarin native, Cantonese fluent all match the Languages line. |
| investment_approach | Systematic — "multi-factor quantitative model focusing on Greater China technology equities" | **correct** | correct | Systematic is supported: the quote about independently building and managing a multi-factor quantitative model is verbatim, and backtesting systematic strategies is also stated. |
| search_summary | Mid-level systematic equity analyst (2.3y relevant). Covers equities in the technology sector. Markets covered are Greater China and Asia. Most recent employer … | **correct** | correct | v7 parse: backtesting, attribution and the Sharpe 1.8+ model sit with Bank of China, and Python, R and SQL are in a separate skills sentence with no role attached. |
| experience[0] | Bank of China / Investment Analyst / 2021-06→2023-09 / full_time / core | **correct** | correct | Bank of China, Investment Analyst, June 2021 to Sep 2023 parses to 2021-06/2023-09, full-time, core. The firm_type Sell-side bank is an inference the text does not state, but no enum value fits better. |
| experience[1] | Bank of China / Investment Intern / 2019-12→2021-05 / internship / core | **correct** | correct | Investment Intern Dec 2019 to May 2021 is correctly attached to Bank of China, which the text only implies by layout; dates 2019-12/2021-05, internship. |
| experience[2] | Chinese University of Hong Kong / Research Assistant (Economics Area) / 2019-03→2021-03 / academic / adjacent | **correct** | correct | CUHK Research Assistant 2019-03 to 2021-03, employment_type academic (not industry experience), relevance adjacent. Reviewed by user. |
| experience[3] | Meridian Asia Capital / Finance Market Department Intern (Chinese Bond and Equity Market) / 2019-07→2019-08 / internship / core | **correct** | correct | Meridian Asia Capital intern Jul to Aug 2019, Jiangxi China, parses to 2019-07/2019-08, country CN, internship. |
| experience[4] | Fujia Aid Teaching Society / Vice President & Head of Public Relations Team / 2019-09→2020-06 / extracurricular / unrelated | **correct** | correct | v7 parse: employer, title and Sep 2019 to Jun 2020 right; the student society post is typed extracurricular, relevance unrelated. |
| experience[5] | University Swimming Club / Head of Promotion Team / 2018-09→2019-09 / extracurricular / unrelated | **correct** | correct | v7 parse: employer, title and Sep 2018 to Sep 2019 right; the student club post is typed extracurricular, relevance unrelated. |

## Zara Al-Rashid  (`5000799aceba`)

| Field | Parsed value | Human | Judge | Reason |
|---|---|---|---|---|
| name | Zara Al-Rashid | **correct** | correct | 'Dr. Zara Al-Rashid' is parsed as 'Zara Al-Rashid'; dropping the honorific is acceptable. |
| contact | {"email": null, "phone": null, "location_text": null} | **correct** | correct | The text carries no email, phone or address (contact is stripped), so all-null is right. |
| education | Indian School of Business Management (ISBM), Pune: PGDM (Masters); K.J. Somaiya Medical College, Mumbai: M.B.B.S (Bachelors); Mithibai College, Mumbai: XII Std.(PCMB) (Other); St. Xavier's High School, Mumbai: X Std. (Other) | **correct** | correct | v7 parse: all four entries with the right years; PGDM is Masters per the reviewer rule, M.B.B.S is Bachelors with field Medicine, XII and X are Other. |
| experience | 6 roles | **correct** | correct | v3 parse records J.P. Morgan as three one-month stints (2013-09, 2013-12, 2014-11), 3 months, per reviewer; all six roles present. |
| skills | {"programming": [], "tools": ["MS Office", "Bloomberg", "Factset", "Reuters", "FDA", "MedPAC", "CMS", "WHO", "German DRG", "USRDS", "PubMed"], "domain": ["Equit… | **correct** | correct | MS Office, Bloomberg, Factset, Reuters and the healthcare databases (FDA, MedPAC, CMS, WHO, German DRG, USRDS, PubMed) all appear under DATABASE SKILLS; domain terms are supported. |
| certifications | [] | **correct** | correct | None are listed. CAT percentile and awards are not certifications. |
| languages | [] | **correct** | correct | No language is listed, so empty is right. |
| investment_approach | Fundamental — "publish equity research ratings with price targets on Indian healthcare sector companies across 28 s…" | **correct** | correct | Fundamental is supported: sell-side equity research with ratings, price targets and 28 stocks under coverage. The quote is verbatim. |
| search_summary | Principal fundamental equity research analyst (10.8y relevant). Covers equities in healthcare, pharmaceuticals and healthcare services. Markets covered: India a… | **correct** | correct | v7 parse: facts supported (28 stocks, 35% at Prism, Bloomberg/Factset/Reuters, PGDM). 10.8y no longer merges the J.P. Morgan stints; the 0.2y shortfall comes from experience[4] being typed internship and is graded there. |
| experience[0] | Meridian Research Partners / Lead Analyst – Pharmaceuticals and Healthcare / 2022-05→now / full_time / core | **correct** | correct | v7 parse: Meridian Research Partners, Lead Analyst, May'22 to till now parses to 2022-05 with is_current true. firm_type is Other; the resume does not name the firm type, and Sell-side broker would fit the ratings and price-target work better. |
| experience[1] | Prism Financial Services / Lead Analyst – Pharmaceuticals and Healthcare / 2020-10→2022-04 / full_time / core | **correct** | correct | Prism Financial Services, Oct-'20 to Apr'22 parses to 2020-10/2022-04. |
| experience[2] | Kotak Securities / Lead Analyst – Pharmaceuticals and Healthcare / 2018-04→2020-04 / full_time / core | **correct** | correct | Kotak Securities, Apr-'18 to Apr-'20 parses to 2018-04/2020-04; the 'Hosted Organized' bullet is verbatim. |
| experience[3] | DataFlow Research India Pvt. Limited / Manager – Equity Research / 2015-05→2018-03 / full_time / core | **correct** | correct | DataFlow Research India, Noida, Manager Equity Research May-'15 to Mar-'18 parses to 2015-05/2018-03, country IN; the US healthcare coverage at Leerink is reflected. |
| experience[4] | J.P. Morgan Securities / Equity Research Associate / 2013-09→2013-09, 2013-12→2013-12, 2014-11→2014-11 / internship / core | **partial** | partial | v7 parse: the three one-month stints and London/GB are right, but employment_type is internship; the text never says intern and she had finished the PGDM in 2012, so full_time. |
| experience[5] | Cipla Pharmaceuticals / Summer Intern / 2011-04→2011-05 / internship / adjacent | **correct** | correct | Cipla summer intern Apr-'11 to May-'11 parses to 2011-04/2011-05, internship, Mumbai; the stray 'Key achievements' header was cleaned out of the bullet. |

## Omar El-Hassan  (`05c2a9b47e80`)

| Field | Parsed value | Human | Judge | Reason |
|---|---|---|---|---|
| name | Omar El-Hassan | **correct** | correct | 'Omar El-Hassan' matches the heading. |
| contact | {"email": "o.elhassan15@gmail.com", "phone": "+ 33678924531", "location_text": null} | **correct** | correct | Email o.elhassan15@gmail.com and phone +33678924531 (stored as '+ 33678924531', with the space the PDF gave) match; location_text null is acceptable since the text only gives a job location. |
| education | INSA Lyon: Diplôme d'Ingénieur (Masters); Universite Lyon: Master's degree (Masters); École Royale Navale: Preparatory Classes (Other) | **correct** | correct | INSA Lyon Diplome d'Ingenieur Sep 2019 to Sep 2022, Universite Lyon Master's Sep 2021 to Sep 2022, and the Casablanca preparatory classes Sep 2017 to Jul 2019 are all present with correct dates and countries. |
| experience | 3 roles | **correct** | correct | BNP Paribas full-time, the BNP internship and the Societe Generale internship are all present. The academic projects are not roles and appear only in skills. |
| skills | {"programming": ["C++", "C#", "Python", "Java", "R", "SQL", ".NET"], "tools": ["MongoDB", "Excel"], "domain": ["Machine Learning", "Stochastic calculus", "Proba… | **correct** | correct | v7 parse with skill dedup: every skill is in the text, C# appears once, and MongoDB is filed under tools per the skills taxonomy. |
| certifications | [] | **correct** | correct | The resume lists none. |
| languages | [{"name": "English", "level": null}, {"name": "Arabic", "level": null}] | **correct** | correct | English and Arabic are listed with no level; the parse matches. French is not listed. |
| investment_approach | Unclear — "Development and back testing of momentum-based trading strategies" | **correct** | correct | v3 parse labels Unclear. Reviewer rule: approach derives only from full-time core/adjacent roles; the only evidence is an internship bullet, so Unclear is the right label. |
| search_summary | Mid-level quantitative developer with unclear investment approach (4.4y relevant). Asset classes are fixed income and derivatives. Currently at BNP Paribas CIB … | **correct** | correct | v7 parse: claims check out against the text (BNP Paribas CIB quantitative developer, Gaussian copula model, local spread option, scenario analysis and Greeks in the C++ pricing library, INSA Lyon). The approach is stated as unclear, matching the approach field, and 4.4y counts only the full-time BNP role. |
| experience[0] | BNP Paribas CIB / Quantitative Developer / 2022-05→now / full_time / core | **correct** | correct | BNP Paribas CIB Quantitative Developer, Paris, since May 2022, parses to 2022-05 with is_current true; bullets match the PDF's scrambled column text. |
| experience[1] | BNP Paribas CIB / Internship / 2022-04→2022-08 / internship / core | **correct** | correct | The 'Internship - April 2022 to August 2022' line parses to 2022-04/2022-08 under BNP. The source text overlaps this with the 'Since May 2022' full-time role, so the parse reproduces an inconsistency in the resume, not a parser error. |
| experience[2] | Société Générale - SGATS (SGCIB) / Quantitative Strategist - Intern / 2021-06→2021-08 / internship / core | **correct** | correct | Societe Generale SGATS, Quantitative Strategist Intern, Rabat, June to August 2021 parses to 2021-06/2021-08, country MA, internship. |

## Location (all candidates)

Gold in `location_human.jsonl`; derived is `current_country` in the database.

| Candidate | Gold | Derived | Basis |
|---|---|---|---|
| Chen Li | **HK** | HK | No contact location. Latest entry is the Bank of China role in Hong Kong (to Sep 2023), just after the HKU MSc (Jul 2023, Hong Kong). |
| Marcus Chen-Rodriguez | **US** | US | No contact location. Current role at Coatue Management is in the US. |
| Marina Silva Costa | **US** | US | No contact location; the +44 phone is not a stated location. The current Vanguard role gives no location, so the latest entry with one is the MIT Sloan MBA (to May 2021, US), after Bain in Brazil (to 2019). |
| Michael Rodriguez | **US** | US | Contact address 'Boston, MA 01125'. |
| Omar El-Hassan | **FR** | FR | No contact location. Current role at BNP Paribas CIB is in France. |
| Priya Nakamura | **IN** | IN | No location in the contact block, any role or any school. Falls back to the head office of the current employer, ICICI Securities (India). |
| Ryan Patel | **US** | US | Contact address 'Brooklyn, NY 11201'. |
| Vikram Shah | **US** | US | No contact location. Current role at Cinctive Capital Management is in the US. |
| Viktor Sharat | **IN** | IN | No contact location and no role gives a location or dates. Latest school is Mumbai Institute of Management Studies and Research (2015, India). |
| Zara Al-Rashid | **IN** | IN | No contact location. Meridian, Prism and Kotak give no location; the latest entry with one is DataFlow Research in Noida (to Mar 2018, India). |
