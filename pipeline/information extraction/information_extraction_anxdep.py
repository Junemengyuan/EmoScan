from openai import OpenAI
from tqdm import tqdm
import os
import json
import pandas as pd

# get the OpenAI api

openai_key = "your api_key"
client = OpenAI(api_key=openai_key)

def response_return(system_role_,prompt_):
    completion = client.chat.completions.create(
    model="gpt-4",#
    messages=[
        {"role": "system", "content": system_role_},
        {"role": "user", "content": prompt_}
    ]
    )
    return completion.choices[0].message.content


# get the data

os.chdir('your path')
case_test_file_anxdep = pd.read_excel("anxdep_cases_v2.xlsx")
case_test_file_augmentation = pd.read_excel("anxdep_testingdata_augmentation.xlsx")

# prompt to fill in the psychiatric profile

system_role_ = f"""Read following case description and the case's symptoms that may be related to the diagnosis, fill in the section of a psychiatric evaluation for the client if there is any related information. Information in the parathesis in the evaluation template are for reference, please do not include those texts (in parathesis) in output. If there is no related information for specific section, fill the section with "None".
    """

format_prompt_ = f"""The psychiatric evaluation template is like:
Identification (Age, marital status, sex, where born, number of siblings, occupation, language if other than English, ethnic background, religion, current living circumstances if pertinent; whether referred or brought by someone): xxx
Chief Complaint (Patient’s own words in quotes): xxx
History of Chief Complaint (Chronological picture of the events that brought the patient to treatment, onset of current episode, precipitating factors): xxx
Past Psychiatric History (stressors, onset, symptoms, course, first episode of illness, types of treatments, hospitalizations, effects on patient, suicidal attempts (includes methods, consequences, drugs or alcohol associated? Seriousness), assaultive behavior, psychiatric medications): xxx
Medical History (Allergies, review of symptoms (includes changes in appetite, head injury, chronic pain, unconsciousness, premenstrual syndrome, review for somatization disorder), any major medical illnesses, surgeries, physical impairment, adult physical or sexual abuse, head trauma, tumors, seizures, infectious diseases, sexually transmitted infections, autoimmune diseases, exposure to environmental hazards or toxins, hospitalizations, recent international travel; include current medications, appetite, sleep pattern, sexual behavior): xxx
History of Drug or Alcohol Abuse (Description of pattern of alcohol and/or drug [illicit or prescribed] abuse: types of substances, duration of use, quantity, consequences (includes medical problems, loss of control, personal and interpersonal problems, job difficulties, legal consequences, financial problems). Misuse of medications (includes prescription, over-the-counter). Include any history of substance- related blackouts or seizures or any intravenous drug use.): xxx
Family History (Psychiatric illnesses, hospitalizations, substance abuse, home environment, relationships, include genogram when appropriate): xxx
Personal History
Perinatal (Perinatal exposure to alcohol or drugs, full-term birth, vaginal or C-section, breast or bottle-fed): xxx
Childhood (Developmental milestones, education, hobbies, interests, history of head banging, rocking, attachment history, separation anxiety, gender identity development, friendships, intellectual and motor skill, learning disabilities, nightmares, phobias, bedwetting, fire setting, cruelty to animals): xxx
Adolescence (problems related to puberty abuse (physical or sexual), education, hobbies, interests, school groups, activities, sports, sexual activity, self- esteem, body image): xxx
Adulthood (Education, employment, marital or relationship issues, children, sexual preference, adjustment, cultural, religious, leisure activities, social, military, legal issues): xxx
Trauma/Abuse History (Describe characteristics of exposure to trauma, such as violence, sexual abuse, military trauma in combat, medical trauma): xxx

MAKE SURE!! Do not include the texts in parathesis in output!!! If there is no related information for specific section, fill the section with "None".

You can refer to this example (but replace the term "patient" with "client" in your output): {case_test_file_anxdep["case description"][72]}
"""

# extract information to generate the psychiatric profile

case_descript_ls = []
for i in tqdm(range(0,len(case_test_file_anxdep),1)):
    #get the important symptoms mentioned in the explanations
    system_role_symptoms = "Read the following discussion on a case, provide the symptoms of the client. Don't contain diagnosis. Don't contain any information that is not in the discussion."
    prompt_temp = case_test_file_anxdep["modified_explanations"][i]
    temp_symptoms = response_return(system_role_symptoms, prompt_temp)
    #input the case description and symptoms, output the outline
    output = response_return(system_role_, " Case description: " + case_test_file_anxdep["case description"][i] + " Case's symptoms that may be related to the diagnosis: " + temp_symptoms + format_prompt_)
    case_descript_ls.append(output)

# save the extracted information

anxdep_save = pd.read_excel("anxdep_cases_v2_save.xlsx")
anxdep_save["extracted_info"] = case_descript_ls
save.to_excel("anxdep_cases_v2_save.xlsx", index = False)
