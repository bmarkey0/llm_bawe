"""
Use vLLM to obtain Llama chat responses.
"""

import argparse
import os.path
import os
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from vllm import LLM, SamplingParams

from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
import gc


cond1 = """
You're a teacher grading a paper. Use a five point scale to grade whether the following paper is organized, and 
demonstrates a developed and sustained order within and across paragraphs. Do not summarize 
or provide feedback. Just give the grade (e.g., level 2, level 4, etc.) where 1 is the least organized and 5 
is the most organized. 

"""

cond2 = """
You're a teacher grading a paper. Use a five point scale to grade whether the following paper is organized, and 
demonstrates a developed and sustained order within  and across paragraphs. Do not summarize or provide feedback. 
Just give the grade (e.g., level 2, level 4, etc.) where 1 is the most unorganized and 5 is the most organized.
"""

cond3 = """
You're a teacher grading a paper. Use a five point scale to grade whether the following paper is organized, and 
demonstrates a developed and sustained order within and across paragraphs. Do not summarize or provide feedback. 
Just give the grade (e.g., level 2, level 4, etc.). Use the following five point scale, where 5 is the highest and 1 
is the lowest:

5: There is a sophisticated arrangement of content with evident and/or subtle transitions. Organizational structure 
is appropriate and strengthens the response and allows for the advancement of the central idea. The sentences, 
paragraphs, and ideas are logically connected in purposeful and highly effective ways. 

4: There is a functional arrangement of content that sustains a logical order with some evidence of transitions. 
Organizational structure is logical and allows for advancement of the central idea. Sentence-to-sentence connections 
are used, but not always effectively. 

3: There is an inconsistent arrangement of content. Transitions attempt to connect ideas but lack purpose and/or 
variety. Organizational structure is inconsistent, does not allow for the even advancement of the central idea. 
An organizational structure that supports logical development is not always evident. 

2: There is a confused arrangement of content. Attempts at transitions are confusing. Organizational 
structure is repetitive, disrupting the advancement of ideas. An organizational structure that supports 
logical development is not appropriate to the task.

1: There is minimal control of content arrangement. There is no central idea. The paper demonstrates little 
or no discernible organizational structure. Transitions are absent. 

"""

cond4 = """
You're a teacher grading a paper. Use a five point scale to grade whether the following paper is organized, 
and demonstrates a developed and sustained order within and across paragraphs. Do not summarize or provide feedback. 
Just give the grade (e.g., level 2, level 4, etc.). Use a five point scale, where 5 is the highest and 1 is the lowest. 
Here are descriptions of points 1, 3 and 5 of the five point scale:

5: There is a sophisticated arrangement of content with evident and/or subtle transitions. Organizational structure 
is appropriate and strengthens the response and allows for the advancement of the central idea. The sentences, 
paragraphs, and ideas are logically connected in purposeful and highly effective ways. 

3: There is an inconsistent arrangement of content. Transitions attempt to connect ideas but lack purpose and/or 
variety. Organizational structure is inconsistent, does not allow for the even advancement of the central idea. 
An organizational structure that supports logical development is not always evident. 

1: There is minimal control of content arrangement. There is no central idea. The paper demonstrates little or no 
discernible organizational structure. Transitions are absent. 

"""

cond5 = """
You're a teacher grading a paper. Use a five point scale to grade whether the following paper is organized, 
and demonstrates a developed and sustained order within and across paragraphs. Do not summarize or provide feedback. 
Just give the grade (e.g., level 2 ,level 4, etc.). Use the following five point scale, where 5 is the highest and 1 
is the lowest:

5: There is a sophisticated arrangement of content with evident and/or subtle transitions. Organizational structure 
is appropriate and strengthens the response and allows for the advancement of the central idea. The sentences, 
paragraphs, and ideas are logically connected in purposeful and highly effective ways. 

4: There is a functional arrangement of content that sustains a logical order with some evidence of transitions. 
Organizational structure is logical and allows for advancement of the central idea. Sentence-to-sentence connections 
are used, but not always effectively. 

3: There is an inconsistent arrangement of content. Transitions attempt to connect ideas but lack purpose and/or 
variety. Organizational structure is inconsistent, does not allow for the even advancement of the central idea. 
An organizational structure that supports logical development is not always evident. 

2: There is a confused arrangement of content. Attempts at transitions are confusing. Organizational 
structure is repetitive, disrupting the advancement of ideas. An organizational structure that supports 
logical development is not appropriate to the task.

1: There is minimal control of content arrangement. There is no central idea. The paper demonstrates little 
or no discernible organizational structure. Transitions are absent. 

Here is an example of a high scoring paper:
When comparing the two poems "Her First Week" and "The Spirit is too Blunt an Instrument", it becomes clear that although both deal with a description of a baby and thus have the same topic in general, many differences can be found when analysing them in detail. These differences are reflected in the use of pronouns, in the lexis and in the grammatical structure of the poems, and thus this essay is going to draw a comparison between the two poems by focusing on the above mentioned aspects. Visually, the overall structure of the two poems is very similar. Both contain almost the same number of lines (28 lines in "Her First Week" and 27 lines in "The Spirit is too Blunt an Instrument"), whereas the latter is further subdivided into three different stanzas of nine lines each. The lines neither rhyme nor is there a regular rhythmic pattern which may be due to the fact that the number of syllables in each line varies very much. Thus, both poems contain lines with relatively few syllables (for example with seven syllables in poem C (l.9) or even only three in poem D (last line)) and also very long lines with up to 15 syllables. (line 28 in poem C). What is also similar is that both poems have a lot of enjambments as the sentences are very complex and go on in the next line most of the time. In contrast to the quite similar visual pattern, the use of pronouns is very different in the two poems. Whereas in "Her First Week" over 30 pronouns are used, the pronoun "her" being the predominant one as it occurs 18 times throughout the whole poem, there are only three pronouns in "The Spirit is too Blunt an Instrument". Already the first word in the heading of "Her First Week" is a pronoun and the reader or the audience automatically ask the question to whom this pronoun may relate. But instead of providing them with an answer by referring to a proper noun (like Maria) or a common noun (like the baby) in the first sentence of the poem, two new pronouns occur there: she and I. Thus, it can be guessed that the poem deals with at least two people and it furthermore becomes clear that it is told out of the perspective of one of them as there seems to be an I-narrator or a first person point of view. Although neither in the first sentence nor in the whole poem the word "baby" comes up, the context makes clear that the pronouns "her" and "she" must refer to a baby. In the first sentence for example, the adjective "small" complements the pronoun "she" and combined with the words "first week" and "crib" an indirect reference to a baby is established. Furthermore the "her" in the second line is post-modified by three appositions that describe the baby´s exact position in the crib ("face down in a corner") and the source it might have come from. ("limp as something gently flung down, or fallen from some sky an inch above the mattress." (l. 2/3) ) Contrary to this detailed description of the baby, the "I" in the poem is not further specified. From the first sentence, it becomes clear that it must be an older person as he/she "scan[s] the crib". But neither the age, nor the gender or the relation to the baby (if it is a relative like a brother or mother or someone alien) can be determined, because the referential noun for the pronoun is missing. Only the role, the taking care of the baby, is described in further detail in the following lines which shows that the role of the person is more important than his/her identity. Finally, the identity of the "I" is given away in the final lines of the poem by the expression "when I fed her [...] and offered the breast, greyish-white, and struck with minuscule scars like creeks in sunlight". (l. 25-27) Here, the "I" is gendered female with the help of the phrase "to offer the breast". The breast is the noun head and the following adjective "greyish-white" together with the coordinate clause introduced by the connector "and" form the post-modifier of the head. They provide the information about the person´s age as the greyish-white colour and the wrinkles which are circumscribed with a comparison allude to the fact that the person must be quite old and thus is probably the mother or grandmother of the baby. In contrast to this extensive use of pronouns in "Her First Week", there are only three pronouns in "The Spirit is too Blunt an Instrument" and each time they are near to their referential nouns. Thus, the "their" in line 6 refers back to the "blind bones", the "its" in line 12 refers back to the "ear" two words before and the "their" in "their pain" in line 27 has even three referential nouns which are connected by "and" in "love and despair and anxiety". This last pronoun is particularly important because otherwise, the sense of the sentence would be different. If the pronoun were omitted, the sentence would be: "It is left to the vagaries of the mind to invent love and despair and anxiety and [...] pain." In this construction, four different feelings would be invented. But with the pronoun "their", the structure is more complex as only three feelings are invented (love, despair, anxiety) and each of them is accompanied by pain. By this, even something positive as love gets a negative side and the critical attitude of the persona towards the mind (which is also indicated by the word choice "invented") that is maintained through the poem is underlined. Instead of circumscribing the baby with pronouns as in "Her First Week", there is already a direct reference in the first sentence of poem D: "this baby" (l. 2). What is striking about this expression is the determiner "this" as this implies that the audience already knows the baby or that they can see it. It seems as if the persona would point with his/her finger to (at) the baby. Thus, the use of "this" tries to establish closeness between the baby and the audience whereas the use of "that" would have kept more distance between them. But this closeness is required because the persona who is not an I-narrator this time but who directly addresses the audience wants them to observe the baby in detail. This is indicated by the imperative "Observe the distinct eyelashes..." (l. 10). As the baby becomes an object of observation then, it has to be near to the audience as otherwise the observation cannot be successful. Especially as the things to observe, "eyelashes" and "fingernails" (l.10/11) for example, are all very small. Similar to "The Spirit is too Blunt an Instrument", where the baby is treated like a scientific object that has to be studied, the baby or rather the baby´s body parts in "Her First Week" also are the objects of the sentences most of the time. This can be seen from the fact that the pronoun "her", as already mentioned above, is the most predominant pronoun throughout the poem and that "her" describes an object position - it is either a direct object alone as in "I fed her" (l. 24) or part of a direct object as in "I would tuck her arm" (l.5). Consequently, the baby is in a passive and helpless role whereas the I, which is in subject position, takes over the active part by caring for the baby. Thus, by using many transitive verbs with 'I' as subject and 'her' as object for describing actions between them, the active/passive relation is foregrounded. In general, the process of how the 'I' takes care of the baby is described with the help of complex sentences and in a very detailed manner: "I would tuck her arm along her side and slowly turn her over. She would tumble over part by part, [...],I´d slip a hand in, under her neck, slide the other under her back, and evenly lift her up." (l. 4-11) Here, the first sentence is a coordinated one which is connected by the connector "and". The second sentence is even more complex as it consists of three parts with three different phrasal verbs (slip in, slide under, lift up) which all refer back to one subject, the I. By connecting the different phrases with commas and "and", a chronological order can be established that shows what is done first, second, third... Furthermore, the use of two adverbs of manner, "slowly" and "evenly", and three adverbials to determine place ("along her side", "under her neck", "under her back") display how carefully the I treats the baby. Every movement seems to be planned and done with great care so that nothing will happen to her. How precious the baby is for the I, is also reflected by the modality of the sentences. By using the auxiliary verb "would" and thus the conditional, the persona shows that she describes a process she seems to have done many times before and which she has already internalized. This also fits with the expression "Every time I checked" (l. 21). By underlining the baby´s fragility and helplessness with the help of sentence structure and word choice (as adjectives like "small" (l.1), "little" (l.11) and "limp" (l.2) show), the persona´s worries that the baby could die become more understandable. It also seems as if the baby plays a very important role for the humanity as whole (l.22) and the high responsibility of the I is expressed with the metaphor: "as if, history of the vertebrate had been placed in my hands". (l. 20) Besides this metaphor, the baby´s weakness and need for help is also foregrounded with the help of several comparisons. By stating that she is "like a load of damp laundry in the dryer" (l. 8) or "like a loose bouquet to my side" (l. 25), it is shown that the baby does not have any control over herself yet and that it needs somebody who nourishes it similar to flowers who have to be watered. Contrary to this really passive role throughout the poem, she becomes active in the last line. The expression "she was willing to stay" attributes her an own will and the desire to be alive. The use of a progressive form, which contrasts to the simple tense in the rest of the poem, furthermore underlines that this will to stay is permanent and will not fade away. In contrast to "Her First Week", where the I is really concerned about the baby and its health, the persona in "The Spirit is too Blunt an Instrument" only sees the baby, as already mentioned above, out of a scientific angle. This can be seen by the highly technical vocabulary in lines 5-16 (for example "ossicle" (l. 14) or "capillaries" (l. 15)) which is used to describe the different body parts of the baby or as said in the poem "the intricate exacting particulars" (l. 5). Besides the two pre-modifying adjectives here, all the rest of the stanza serves as a post-modifier of the common noun "particulars" by describing in detail what these particulars are: "the tiny blind bones with their manipulating tendons, the knee and the knucklebones, the resilient fine meshings of ganglia and vertebrae in the chain of the difficult spine." (l.5-9) Structurally, this is not a real sentence as the verb is missing, but only a list of noun heads with their respective modifiers connected by "and" and commas and glued together with the help of many prepositions. The effect of this enumeration and parallel structure is that the persona can emphasize all the different aspects of the body and thus underline its complicated structure. Furthermore, the repetitive use of "the" (about 15 times in the whole poem) instead of "a" is probably used to establish a feeling of familiarity with the different parts as, though not thinking of them, everybody uses them unconsciously most of the time. The listing then goes on in the whole next stanza and thus this stanza only consists of two complex sentences, both initiated by an imperative. Consequently, both sentences lack a subject. With the help of the imperatives, the persona directly addresses the audience and wants them to "observe" (l. 10) and "imagine" (l. 14) all these great details that the small body of a baby already has. By this, he/she follows the aim to persuade the audience of a thesis he/she has, namely that "The spirit is too blunt an instrument to have made this baby." (line 1-2) Thus, in contrast to the descriptive sentence style in poem C, the persona in poem D tries to convince the audience by building up an argumentative structure. The thesis in the first two lines is followed by the main argument, namely that a baby´s body is already so perfect that it cannot only be the result of "human passions" (l. 3). In that line, the conditional modality of the auxiliary verb 'could' in "could have managed" displays the hypothetical nature of the sentence. The listing then serves as the main argument as it displays the complexity of the baby´s body. Here, the pre-modifying adjectives are mainly used to emphasize this complexity and perfection even more as can be seen in "flawless" (l. 15) or "difficult" (l. 9). By this, the persona wants to make the audience share his/her opinion. The persona´s conclusion is then introduced and highlighted by the exclamation "No." in line 21. This expression is structurally prominent as, lacking noun and verb, it is no real sentence. By standing on its own, it displays a complete negation without any compromises on the persona´s side. The repetition of "no" in the next sentence then explains in further detail to what the persona opposes so much. The use of the conditional in "could have done" displays here again the hypothetical character of the sentence. All in all, though the two poems both deal with the description of a baby, they are very different. The poem "Her First Week" describes the relationship between a weak, little baby and a loving person taking care for her and thus contains many verbs which describe the action between the two. Here, the mother or grandmother is mainly the agent, which is displayed by the subject position of the I, whereas the baby is in object position most of the time because of her role as a patient. The choice of words furthermore shows how concerned the I-narrator is with the baby´s health. In contrast to this loving relationship, the persona in "The Spirit is too Blunt an Instrument" is not interested in the baby itself but he/she deals with it in a scientific manner and only uses it as a proof for his/her hypothesis. Thus, the baby is reduced to its physical perfection and as a consequence, the poem contains more nouns than verbs as all the different body parts with their flawlessness are enumerated. Instead of a descriptive style as in "Her First Week", the lexis and the structure of the sentences in that poem try to persuade the audience to share the persona´s view. 
"""

cond6 = """
You're a teacher grading a paper. Use a five point scale to grade whether the following paper is organized, and 
demonstrates a developed and sustained order within and across paragraphs. Do not summarize or provide feedback. 
Just give the grade (e.g., level 2, level 4, etc.). Use the following five point scale, where 5 is the highest and 1 
is the lowest:

5: There is a sophisticated arrangement of content with evident and/or subtle transitions. Organizational structure 
is appropriate and strengthens the response and allows for the advancement of the central idea. The sentences, 
paragraphs, and ideas are logically connected in purposeful and highly effective ways. 

4: There is a functional arrangement of content that sustains a logical order with some evidence of transitions. 
Organizational structure is logical and allows for advancement of the central idea. Sentence-to-sentence connections 
are used, but not always effectively. 

3: There is an inconsistent arrangement of content. Transitions attempt to connect ideas but lack purpose and/or 
variety. Organizational structure is inconsistent, does not allow for the even advancement of the central idea. 
An organizational structure that supports logical development is not always evident. 

2: There is a confused arrangement of content. Attempts at transitions are confusing. Organizational 
structure is repetitive, disrupting the advancement of ideas. An organizational structure that supports 
logical development is not appropriate to the task.

1: There is minimal control of content arrangement. There is no central idea. The paper demonstrates little 
or no discernible organizational structure. Transitions are absent. 

Here is an example of a low scoring paper:
As the title already indicates, the poem "Not a Nice Place" describes a violent place, probably a whole country, where people live in danger and have to fear for their lives every day. The mood of chaos and violence in the poem is established with the help of different linguistic features, namely the lexis and word combination, the sound and visual patterning and the structure. Thus, this essay is going to analyse the poem with respect to the above mentioned features. The first two lines of the poem "Holy cow! You should see that place." make clear right from the beginning that there is a persona directly addressing its audience. This is be done with the pronoun "you" and as this can be either singular or plural, there may be either one addressee or several. Thus, although the poem is a monologue, a situation like in a dialogue is established because there is more than one participant. The main part of the poem then (l. 3-26) contains a detailed description of the place hinted to in line 2 and thus refers back to that line. Here, the persona takes over the role of an observer and informs the addressee of the violence and danger people live in. Thus, the favoured sentence-type is a stating one and the third person plural is predominant, either as personal pronoun "they" (l.8) or in subjects like "children" (l.3), "men" (l.6), "women" (l.7) and "people" (l.5). This shows that everybody, even children, is involved in the conflict. The last two lines are similar to the first two as they involve a direct address to the audience via a rhetorical question. By this, a frame is created which underlines the persona´s shock ("Holy cow!" (l.1) ) about what he/she sees and which - in contrast to the mainly descriptive main part - displays his/her personal judgement of the events. Concerning the visual and sound pattern, it is very striking that the poem does not fulfil many of the typical features of its genre poetry. It is neither subdivided into stanzas, nor is there a particular rhyme scheme. There is also no regular rhythm or metre. The only visual aspect that is maintained is that the poem is written in lines. But unlike other poems, where every line begins with a capital letter, no matter if the line starts in the middle of a sentence, some lines also begin with small letters here. They are only capitalized if they mark the beginning of a new sentence. Furthermore, the normal sentence structure of SVO is maintained and there is no deviation from it. To my mind, all this has the effect that the poem resembles more a dramatic speech spoken to an audience. This also fits to the findings of the first paragraph, namely that there is a persona (or speaker) addressing an addressee (audience) to inform and make them think about the miserable conditions the people have to live in. These terrible conditions are mainly illustrated by the lexis and the combination of words. The vocabulary is predominantly from the area of war as for example the words "bulletproof" (l.3), "war surgeons" (l.11), "shot" (l.12) and "Uzi sub-machine guns" (l.23) show. By this, an atmosphere of chaos and danger is established. The use of parallelisms underlines this impression. In particular the parallelism in lines 12-16 supports the image of violation of law as you can get killed "for your Rolex watch/or your Nike boots/ or if you look crooked at somebody" (l.13-15). In this parallelism, the reason for getting shot becomes less and less important. First you are shot for the expensive things you wear -"Rolex" and "Nike" serve as symbols for expensive branches (Rolex is even more expensive than Nike) here - but then even for the way you look at other people. Thus, the parallelism stresses the senseless- and arbitrariness of the crime. Another aspect, which is graphically salient because of the inverted commas, is the use of direct speech in line 21. By foregrounding the "Have a nice day", the people's ignorance and attempt to live a normal life is contrasted to the harsh reality that is described in the rest of the poem. The sentence has also an ironical connotation as you will never be able to have a nice day in such a surrounding. Besides the word choice and combination, the phonology of the words also helps to draw a picture of a violent and dangerous place. Throughout the poem, many plosives - especially the bilabial phoneme /p/ - are used. As these are very short and aspirated sounds, they sound like single gun shots and thus support the image of war. A very good example are the "bullet proof back packs" (l.3) as - when read aloud - it really seems as if bullets were shot with a machine gun because of the /b/ /p/ /b/ /p/ sequence. Another example is "could ge t sho t" (l.12) where the combination of the plosives /k/ and /t/ with monosyllabic words produces disjoint, machine gun like sounds. Besides the plosives, the use of the /oi/ and /ei/ diphtongs in "oi vei" (l. 12) helps to create a sound of deep sorrow to display the persona's grief about the situation. This is because diphthongs are very long sounds as they glide from one vowel to another and they can be drawn out by a speaker as long as he/she wishes. Another sound pattern is the alliteration in the title "Not a Nice Place" that is also repeated in line 10. By foregrounding the "not nice" phonologically, the euphemism of that expression is underlined as the place is not only not nice but rather like hell. For my rewrite, I have also chosen this line and I will rewrite it from "It is not a nice place" to "Oh, it's such a lovely place." The reason for this rewriting is that people often, when asked about their home country and if they like it there, answer something similar to the rewrite. But in this context, such an answer is completely inappropriate as the place described is not a place that anybody would call "lovely". Thus, with the help of the sharp contrast between this line and the rest of the poem, the addressee realizes the irony of the sentence and its sarcastic connotation. To my mind, the rewritten line is even more salient than the original one as the use of the word "lovely" creates a greater deviation from the general mood and lexis of the poem than does "not nice". Because of the ironic connotation, the persona's criticism and attempt to dissociate himself/herself from the described place becomes even clearer. All in all, the fact that a persona directly addresses the audience like in a dramatic speech has the effect that the poem's content is brought closer to the addressee as they get more involved. Furthermore, by combining war vocabulary with such phonological effects as described in the essay, the danger and violation of law can be stressed effectively as these are perceived visually as well as audibly. Thus, each linguistic feature that has been described above contributes to the main topic of the poem, namely to draw a picture of a violent and dangerous place and to criticize the behaviour of the people living there. 
"""


cond_list = [cond1, cond2, cond3, cond4, cond5, cond6]

def get_prompts(sourcefile):
    prompts = pq.read_table(sourcefile).to_pandas()

    return prompts["filename"].tolist(), prompts["text"].tolist()


def prepare_prompt(text, tokenizer):
    # matching vllm.entrypoints.chat_utils.apply_chat_template()'s logic
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": cond_list[args.condition]},
            {"role": "user", "content": text}
        ],
        tokenize=False,
        add_generation_prompt=True
    )

def get_completions(texts, model, temp, tensor_parallel_size=2):
    sampling_params = SamplingParams(logprobs=3, top_p=0.9,
                                     max_tokens=750, repetition_penalty=1.0,
                                     temperature=temp)

    print("Initializing LLM")
    # default gpu_memory_utilization of 0.9 causes OOMs
    llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size,
              gpu_memory_utilization=0.8)

    # The LLM.chat() method only supports one chat at a time, and hence is much
    # slower because it does not batch the prompts. We can do it ourselves instead.
    tokenizer = llm.get_tokenizer()

    prompts = [prepare_prompt(text, tokenizer) for text in texts]

    outputs = llm.generate(prompts, sampling_params=sampling_params)

    destroy_model_parallel()
    destroy_distributed_environment()
    del llm.llm_engine.model_executor
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return [o.outputs[0].text for o in outputs], [o.outputs[0].logprobs for o in outputs]

def get_logprobs(logprobs_output):
    big_lst = []
    for i in range(len(logprobs_output)):
        small_lst=[]
        for n in range(len(logprobs_output[i])):
            for key in logprobs_output[i][n]:
                if 16 <= key <= 20:
                    small_lst.append(str(f"{n}: {logprobs_output[i][n][key]}"))
        big_lst.append(small_lst)
    return big_lst

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a chat model and generate texts.")
    parser.add_argument("--size", type=int, help="Maximum number of prompts to run.")
    parser.add_argument("--parallel", type=int, default=2,
                        help="Number of GPUs to use in parallel.")
    parser.add_argument("model", help="Model name (e.g. meta-llama/Meta-Llama-3-8B-Instruct)")
    parser.add_argument("prompts", help="Path to Parquet file containing prompts.")
    parser.add_argument("outdir", help="Directory to save output Parquet file in.")
    parser.add_argument("--condition", type=int, help="Prompt condition for system prompt.")

    args = parser.parse_args()
    model_name = os.path.basename(args.model)

    print(f"Loading prompts from `{args.prompts}`")
    doc_ids, prompts = get_prompts(args.prompts)

    if args.size is not None:
        prompts = prompts[:args.size]
        doc_ids = doc_ids[:args.size]

    print(f"{len(prompts)} prompts loaded")
    print(f"Generating from {args.model}")

    temp_lst = [0, 1.0, 1.5]
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    for item in temp_lst:
        out, probs = get_completions(prompts, args.model, item, tensor_parallel_size=args.parallel)

        final_probs = get_logprobs(probs)

        out_table = pa.table([doc_ids, out, final_probs], names=["doc_ids", model_name, "probs"])

        pq.write_table(out_table, os.path.join(args.outdir, model_name + f"#Essay_org_{item}.parquet"), compression="gzip")

