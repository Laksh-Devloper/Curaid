# community/management/commands/flood_forum.py
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from community.models import ForumCategory, ForumTopic, ForumReply
import random

User = get_user_model()


class Command(BaseCommand):
    help = 'Floods the forum with unique, sarcastic, adult-friendly health content'

    def handle(self, *args, **options):
        bot_users = list(User.objects.filter(email__contains='curaid-bot.com'))
        if not bot_users:
            self.stdout.write(self.style.ERROR('No bots! Run: python manage.py create_forum_bots'))
            return

        data = {
            'General Health': [
                {
                    'title': 'Doctor said I was totally fine. I feel like death. Very cool, very normal.',
                    'content': 'Went in with 12 symptoms. Got 2 tests. Everything came back "in normal range." They handed me a pamphlet on drinking water and sent me home. Meanwhile I am dragging myself out of bed every day like a sad Roomba with 3% battery. Is this just what being an adult feels like now? Has anyone actually pushed back hard enough to get real answers or do we all just silently suffer?',
                    'replies': [
                        'Welcome to the club. Membership perks include brain fog and being told to reduce stress. There is no exit strategy.',
                        'Chronic fatigue took me three doctors and two years to get taken seriously. Be annoying. It genuinely works.',
                        'I tracked 60 days of symptoms in a spreadsheet and printed it out for my appointment. Suddenly they had time for me.',
                        '"Normal range" means you have not died yet. Optimal range is a completely different conversation your GP is not having.',
                        'Get thyroid, iron, B12, and vitamin D checked specifically. Standard panels skip half of what actually matters.',
                    ]
                },
                {
                    'title': 'I googled my symptoms at 2am and apparently I have 4 cancers and a haunting.',
                    'content': 'Slight headache: brain tumor. Tired after lunch: diabetes plus heart failure. Knee clicks occasionally: bone disintegration underway. I know I should stop. I keep not stopping. Does everyone do this spiral or is it just those of us with functioning imaginations and poor impulse control? Looking for solidarity and also a logical explanation for why I keep doing this to myself.',
                    'replies': [
                        'My rule: if WebMD says cancer, it is dehydration. Has been true 100% of the time so far. Zero exceptions.',
                        'I banned myself from health googling after midnight. Single most effective health intervention of my life.',
                        'WebMD shows rarities first because rare = clicks. Your headache is not a glioblastoma. Drink water. Go to sleep.',
                        'Health anxiety is a genuine condition and it is exhausting. A therapist who specializes in this helped me more than any symptom checker.',
                    ]
                },
                {
                    'title': 'Hot take: 8 hours of sleep did more than $800 of supplements ever did.',
                    'content': 'I have spent probably 800 dollars on supplements over three years. Ashwagandha, lions mane, magnesium glycinate, a frightening quantity of vitamin D. You know what actually changed my life? Getting consistent sleep for one month straight. I feel genuinely stupid about this. Someone should have led with that. I am reporting the supplement industry to whoever regulates this.',
                    'replies': [
                        'Sleep is free and more effective than anything sold in a brown bottle with a leaf logo. Nobody profits from you sleeping so nobody markets it.',
                        'Hard disagree on magnesium though. Getting my levels up actually fixed my sleep quality AND muscle cramps. Both at once.',
                        'The issue is all supplements are placebo-amplified by bad sleep. Fix sleep first. Then reassess what you actually need.',
                        'I was taking melatonin to sleep better instead of just going to bed earlier. The answer was embarrassingly simple and I refused to see it for two years.',
                    ]
                },
                {
                    'title': 'My fitness tracker says I burned 3000 calories hiking. I weigh 68kg. This is fiction.',
                    'content': 'Did a moderate 2-hour hike. Watch told me I burned 3247 calories. Reader, I then ate an entire pizza because I had earned it. The math did not math. Now I understand why my weight loss stalled. Are smartwatches actually useful for anything beyond step counting or are we all just wearing expensive pedometers and lying to ourselves about pizza?',
                    'replies': [
                        'Consumer wearables overestimate calorie burn by 20 to 90 percent depending on activity. The pizza was a trap and the watch set it.',
                        'I turned off calorie tracking on mine entirely. Just use it for steps and heart rate. Everything else is creative writing.',
                        'The only accurate calorie measurement is a metabolic lab test and even those have a margin of error. We are all vibing.',
                        'Your watch wanted you to have the pizza. This is a hardware feature, not a bug.',
                    ]
                },
                {
                    'title': 'Water. I know water is good. I cannot drink enough water. Why am I like this.',
                    'content': 'I fill a 1L bottle every morning with full conviction and moral clarity. By 3pm I have had two sips and one coffee. I know hydration matters. I know this. The water just sits there. Mocking me. There is no flavour reward so my brain refuses to engage. What actual dark sorcery are you using to hit your hydration goals because willpower absolutely does not work and I have tried it repeatedly.',
                    'replies': [
                        'Sparkling water changed everything for me. It feels like a treat. I cannot explain it. My brain is easily fooled.',
                        'One glass before every meal. That is three glasses with zero additional effort required. Build on that.',
                        'The 8 glasses rule is not actually scientifically established. Drink when thirsty. Your kidneys are smarter than a 1940s army pamphlet.',
                        'Electrolyte powder makes it taste like something. Technically still water. Nobody can stop you.',
                        'I put rubber bands on my wrist each time I finish a glass. My brain needs gold stars. I accept this about myself.',
                    ]
                },
            ],
            'Mental Health': [
                {
                    'title': 'Therapy is great until your therapist says something so accurate it physically ruins your week.',
                    'content': 'Went in to complain about a coworker. Forty-five minutes later I am crying about my father and realizing I have been recreating childhood power dynamics in every workplace I have ever worked in. I came for validation, Sharon. Not enlightenment. Not growth. This is too much personal development for a Tuesday. Does therapy always feel like getting psychically punched by someone you are paying?',
                    'replies': [
                        'Yes. This is exactly how you know it is working. Welcome to growing as a person. It genuinely sucks while happening.',
                        'My therapist calls it productive discomfort. I call it please stop being so accurate.',
                        'I once left a session unable to make eye contact with my own reflection for three days. Fully healed now though.',
                        'The sessions you dread revisiting are the ones doing the work. The comfortable ones are maintenance at best.',
                        'Sharon connecting those dots in 45 minutes means you were ready for it. That is actually massive progress.',
                    ]
                },
                {
                    'title': 'Does anyone else feel crushing guilt for taking a mental health day? No? Just me?',
                    'content': 'Called in sick because I was genuinely on the edge. Spent the entire day half resting and half convinced I was being deeply lazy and that everyone at work secretly knew I was faking. I was not faking. I was not okay. Why do I need a doctors note for my own brain to justify not working? When did we collectively decide that mental exhaustion does not count as being ill?',
                    'replies': [
                        'The guilt means you are definitely not lazy. Lazy people do not have this internal conflict. They sleep in.',
                        'Your brain is an organ. If your liver needed a day people would not question it. Same logic applies here.',
                        'I started calling it a sick day rather than a mental health day to my employer. Technically accurate. Zero guilt attached.',
                        'The productivity guilt is a symptom of burnout not evidence that you do not deserve rest. It is circular and completely unfair.',
                    ]
                },
                {
                    'title': 'My anxiety has developed anxiety about my anxiety treatment. How did we get here.',
                    'content': 'Started CBT for generalized anxiety. Now I am anxious about whether I am doing the CBT homework correctly. Also anxious about whether my anxiety responses are improving fast enough. Also anxious about bringing any of this up with my therapist in case it means I am doing it wrong. I have created a nested loop of anxiety from which there is no obvious exit. Is there a restart button somewhere?',
                    'replies': [
                        'Anxiety about treating anxiety is so textbook I am almost impressed. You are not broken. You just have a very enthusiastic nervous system.',
                        'Tell your therapist exactly what you just wrote here. They will find this very useful and not at all unusual.',
                        'I spent three months anxious about whether my medication was giving me the correct side effects. Anxiety is creative.',
                        'The meta-awareness is actually progress. You can observe your patterns. That is the necessary first step.',
                        'CBT with anxiety about the CBT is just recursion. You need a base case. Your therapist is probably it.',
                    ]
                },
                {
                    'title': 'Good vibes only is toxic positivity disguised as wellness and I will die on this hill.',
                    'content': 'Every time I try to express that I am struggling and someone responds with just focus on the positive I want to dissolve into the floor. Negative emotions exist for a functional reason. Suppressing them does not make them disappear it makes them fester and emerge sideways at 2am while you are trying to order food and suddenly everything is too much. Can we normalize saying this genuinely sucks right now and that is valid as a complete sentence?',
                    'replies': [
                        'Good vibes only is the thoughts and prayers of mental health. Aesthetically soothing and completely useless.',
                        'Research shows forcing positivity during negative emotions actually increases stress hormones. It is biologically counterproductive.',
                        'The healthiest response to someone struggling is that sounds really hard. No silver lining required. No pivot to gratitude.',
                        'I unfollowed 40 accounts over exactly this. My baseline anxiety dropped measurably within a month.',
                    ]
                },
                {
                    'title': 'I am fine is my default state and I have no idea when I stopped being able to answer honestly.',
                    'content': 'Someone asked how I was at work yesterday. I said great, really good! while currently in the middle of a low-grade existential crisis running on four hours of sleep. I have said I am fine so many times it has become my idle state. I genuinely cannot answer the question honestly without it feeling weird. Is there a middle ground between lying reflexively and emotionally dumping on a near stranger?',
                    'replies': [
                        'Pretty tired but hanging in there is doing significant heavy lifting in my daily vocabulary right now.',
                        'Having a weird week is honest enough to be real and contained enough to not require follow up. Highly recommend.',
                        'You are describing emotional masking which is extremely common in professional settings and worth actual exploration.',
                        'The Japanese have separate words for your public feelings and your real feelings. Both can be valid. The goal is knowing which is which.',
                    ]
                },
            ],
            'Nutrition & Diet': [
                {
                    'title': 'I did 30 days of clean eating. The main outcome was learning that I hate myself.',
                    'content': 'Day 1: healthy eating is empowering. Day 7: okay this is manageable. Day 14: staring at a brown rice bowl at 8pm wondering how my choices led here. Day 30: I ate an entire block of cheese in celebration and it was genuinely one of the most spiritual experiences of my adult life. Did my body change? Marginally. Did my soul vacate the premises by day 20? Absolutely without question.',
                    'replies': [
                        'Marginally is honest. Most 30-day cleanses produce less physical change than simply drinking adequate water daily.',
                        'The cheese at the end is nutritionally valid and emotionally necessary. Dairy contains calcium, protein, and the will to continue.',
                        'Restriction diets work brilliantly until they catastrophically do not. Sustainable eating is 80 percent reasonable choices and 20 percent necessary cheese.',
                        'The spiritual cheese experience is the funniest and most relatable health journey conclusion I have ever personally witnessed.',
                    ]
                },
                {
                    'title': 'Realized intermittent fasting is just skipping breakfast with Latin branding. My mom already knew.',
                    'content': 'Spent three months doing 16:8 IF and tracking it very seriously in a premium app. Mentioned it to my mum. She said so you just skip breakfast like you always have? and I genuinely could not form a counter-argument. Is intermittent fasting metabolically special or did we simply rebrand meal skipping with optimization language and sell it to people who like feeling scientific about their habits?',
                    'replies': [
                        'The research on IF is genuinely interesting. It is less about calorie restriction and more about insulin response intervals and cellular autophagy.',
                        'Your mum is the most qualified nutrition expert in this scenario and frankly in most scenarios.',
                        'I asked my endocrinologist about IF. She said eat real food, eat less frequently, sleep more. She then charged me 150 euros. Worth it.',
                        'The protocol framing helps certain people psychologically. If calling it 16:8 makes you actually follow through where willpower alone does not, the label is functional.',
                    ]
                },
                {
                    'title': 'Does diet actually affect libido or is that also something wellness influencers invented to sell mushroom powder?',
                    'content': 'I have read six separate articles claiming that zinc, maca root, dark chocolate, oysters, pomegranate, avocado, and fenugreek are all natural libido boosters. That is just a grocery list. My sex drive has been low for months and I would genuinely like to address it without spending 80 dollars on a supplement that tastes like soil and has n=40 evidence behind it. Is there actual science here or is this entirely marketing dressed in citations?',
                    'replies': [
                        'Zinc deficiency genuinely does suppress testosterone in men. Get your zinc levels checked before buying anything.',
                        'Chronic undereating crushes libido faster than almost anything else. If you are in a caloric deficit that is almost certainly the primary issue.',
                        'Maca root has some modest evidence. It will not fix a hormonal problem but it does not taste terrible in a smoothie either.',
                        'The biggest nutritional libido suppressors: high processed sugar, chronic alcohol, and caloric restriction. The fix is genuinely unsexy whole food and maintenance calories.',
                        'Dark chocolate increases blood flow via flavonoids and makes you feel good. The science is actually there. Eat the chocolate without guilt.',
                    ]
                },
                {
                    'title': 'Guy at the gym told me to try carnivore diet. He was visibly breathless walking to the water fountain.',
                    'content': 'I did not ask. I was eating a banana. He volunteered that fruit feeds cancer and that he personally eats nothing but beef and eggs. His complexion was genuinely grey. He had the eyes of someone who had not slept since 2022. He was very confident. I am worried about his kidneys specifically. Has anyone tried carnivore? What is the actual experience versus the online mythology?',
                    'replies': [
                        'It is mostly a personality type with some dietary components attached.',
                        'The research on strict carnivore is extremely sparse because removing entire food groups is difficult to study ethically over time.',
                        'The grey complexion was his body communicating important information. The banana was listening.',
                        'I tried it for two weeks out of curiosity. Energy was adequate. No vegetables for 14 days did things to my digestion I will not recover from fully.',
                        'The fruit feeds cancer claim is so thoroughly debunked by actual oncology research that it is physically painful to encounter in the wild.',
                    ]
                },
                {
                    'title': 'Is anyone actually hitting protein targets without scanning barcodes at every meal like a cult member?',
                    'content': 'Every fitness account I follow has become protein propaganda at this point. Protein at every meal. Protein before bed. I tracked mine once and got 65g which apparently means my muscles are actively dissolving. But I do not want to carry a food scale and scan packaging while eating with my family like I have joined a data-driven religious sect. Is there a sane approach that does not require total surrender of spontaneous eating?',
                    'replies': [
                        'One palm-sized protein source at every meal gets most people to adequate without tracking. Not optimal. Adequate. Good enough.',
                        'The bodybuilder targets are overkill for regular people. 1.2g per kg bodyweight is the well-supported standard recommendation.',
                        '65g on 70kg is low if you are training but 20 more grams per day is one added egg. It does not require a scale or a barcode reader.',
                        'Greek yogurt, eggs, and legumes solved my protein problem without any animal muscle whatsoever. Also significantly cheaper.',
                    ]
                },
            ],
            'Fitness & Exercise': [
                {
                    'title': 'Paid for 8 months of gym membership. Went 6 times. Zero regrets. Ask me anything.',
                    'content': 'January: full of hope and new shoes. February: still attending. March: definitely going Saturday. Late March: definitely going next week. August: membership quietly cancelled while I avoided eye contact with the mirror. I am not here to feel shame. I am here to understand why this is a universal human experience and whether there is actually a mechanism to break the cycle. Because January is about to happen again and I have a concerning amount of confidence about it.',
                    'replies': [
                        'The key insight is finding something you do not actively hate versus something you theoretically respect. Most people hate gyms but love the concept.',
                        'I pay for classes rather than open gym access. Cancelling a class feels specific and slightly shameful. Still attending after 18 months.',
                        'The two-minute rule: commit only to changing into workout gear. 90 percent of the time you will go anyway. The resistance is always starting.',
                        'Home workouts are free and can be done at 11pm in whatever you are wearing without anyone watching you struggle with small weights.',
                        'I put my gym bag in the car. The bag being present means my excuses have to work significantly harder.',
                    ]
                },
                {
                    'title': 'Every realistic workout on the internet takes 90 minutes and requires equipment I do not own.',
                    'content': 'Quick full body workout! 45 minutes plus warmup plus cooldown. What is a cable machine. I have one resistance band from 2020 and two cans of chickpeas. Can someone please design an honest workout for people with a 20-minute window, chronic low motivation, and the self-awareness to admit they are not athletes? The fitness content industrial complex needs grounding in actual reality.',
                    'replies': [
                        '20 minutes AMRAP: 10 pushups, 15 squats, 20 jumping jacks. Repeat until time runs out. You are done. Chickpeas unnecessary.',
                        '3 sets of 10 bodyweight squats, pushups, and lunges every morning before the shower. 12 minutes. I have done this for two years. Healthy. No abs. Both acceptable.',
                        'Quick in fitness content means quick relative to a professional athlete training block. This framing is genuinely deceptive.',
                        'The fitness industry profits from complexity. Squats exist. Walking exists. Stairs exist. Use them. Nobody can sell you a stair.',
                        'The chickpea cans are actually usable. You are further along than you think.',
                    ]
                },
                {
                    'title': 'Is walking real exercise or are we all just collectively coping with not running?',
                    'content': 'Doctor suggested more movement. I started doing 8000 steps daily. Sleep is better. Energy is better. Lost 3kg in four months without changing anything else. But every time I tell gym people I exercise they get a very specific look on their face. Is walking cardiovascular exercise by definition or is it what we tell people who cannot run and want a participation trophy?',
                    'replies': [
                        'Walking at brisk pace has equivalent cardiovascular benefit per unit of energy as running. Just distributed across more time. Studies confirm.',
                        'The look on gym peoples faces is the look of someone who has made exercise their personality. You do not need that personality to be healthy.',
                        'Those 3kg are more sustainable than 3kg from six weeks of aggressive training you cannot maintain. This is the actual metric that matters.',
                        'Zone 2 cardio which is mostly brisk walking is where the majority of your aerobic health adaptations actually occur. You are doing the correct thing.',
                        'VO2 max crowd made everyone feel walking is inferior training. It is not. Walking is low injury, sustainable, and effective. It wins.',
                    ]
                },
                {
                    'title': 'Started tracking my resting heart rate. Now I watch it like a volatile stock portfolio.',
                    'content': '58 bpm this morning: excellent, I am thriving. 63 bpm: what did I do wrong, am I dying, was it the wine? I check it before I am fully conscious. I cross-reference it against sleep score, HRV, and a vague sense of cosmic wrongness. Is health tracking actually making me healthier or have I simply transferred my anxiety to a new and more data-rich dashboard?',
                    'replies': [
                        'The anxiety transfer to biometric dashboards is extremely common and exactly as useful as it sounds.',
                        'Night heart rate elevation is mostly: alcohol, undereating, overtraining, stress, and poor sleep. Not sinister. Just annoying.',
                        'I deleted my tracking app for a month after I started WebMD-ing every elevated reading. Health first. Data second.',
                        'Single day variation is noise. 30-day trend is signal. Look at trends, put the phone down, go outside.',
                    ]
                },
                {
                    'title': 'Does exercise actually improve your sex life or is that just gym propaganda with extra steps?',
                    'content': 'Every list of exercise benefits includes improved sex life somewhere between better sleep and more energy. I have been consistently training for four months. I am objectively fitter. I am not certain my sex life has dramatically transformed and I want actual receipts. What is the physiology here? Does it matter whether it is cardio versus strength? Is there a specific type of training that has actual evidence? I want specifics not a magazine headline.',
                    'replies': [
                        'Cardiovascular fitness directly improves sexual stamina because sex is itself a cardiovascular activity. This part has solid evidence behind it.',
                        'Strength training increases testosterone in both sexes which does have a genuine and consistent effect on libido in the research literature.',
                        'Pelvic floor specifically: hip and core functional training improves sexual function measurably in multiple studies. This one is undertalked.',
                        'The confidence component is probably underrated. Feeling comfortable in your body changes how you engage with it. Real psychological effect.',
                        'The effect is real but modest for most people. It will not fix relationship dynamics or actual hormonal issues but it is a genuine contributing factor.',
                    ]
                },
            ],
            'Chronic Conditions': [
                {
                    'title': 'Got my PCOS diagnosis and 4 doctors gave me 4 completely different treatment plans. Fantastic system.',
                    'content': 'Low carb. No wait complex carbs only. Cut dairy. Actually dairy is completely fine. Take inositol. No take berberine. Actually take both. Exercise but not excessively. Reduce stress, laughing out loud, sure. Metformin now. Actually try naturally first. Birth control. Actually birth control may worsen it. I have seen four different specialists. They disagreed on almost everything. I am more confused about my own body than before I was diagnosed.',
                    'replies': [
                        'Inositol specifically the 40 to 1 myo to d-chiro ratio has the most consistent evidence for insulin-resistant PCOS. Start there before anything else.',
                        'Low glycaemic eating not keto not strict low carb but avoiding blood sugar spikes changed my cycle regularity entirely.',
                        'PCOS has four subtypes and what resolves one makes another worse. Getting full hormonal and insulin panels done helped identify which type I actually have.',
                        'Finding an endocrinologist who specifically specializes in PCOS rather than a general GP was the single most useful thing I did. The knowledge gap between practitioners is enormous.',
                        'The stress advice is real but completely useless without practical tools. Cortisol directly worsens every PCOS marker. Managing mental health is metabolic management.',
                    ]
                },
                {
                    'title': 'Migraine people: how do you maintain functional lives? I genuinely need to understand your methods.',
                    'content': 'Third migraine this month. Yesterday I spent 14 hours in a completely dark room unable to tolerate light, sound, or the concept of time. My colleague gets headaches and takes paracetamol and continues working. I need complete sensory deprivation followed by two days of cognitive fog. How are migraine sufferers maintaining careers and relationships? I need an actual management framework because I cannot keep cancelling my life.',
                    'replies': [
                        'Trigger identification changed everything. Mine was the combination of poor sleep, dehydration, and red wine together. Removing the triple trigger cut frequency by 70 percent.',
                        'Triptans taken at symptom onset not when the migraine has peaked are genuinely effective. If your doctor has not discussed these, ask specifically.',
                        'Three per month is clinically significant. That frequency typically warrants preventative medication not just acute management.',
                        'I got Botox injections for chronic migraine. This sentence used to embarrass me. Six months in, two episodes total. I formally apologize to Botox.',
                        'The migraine hangover which is called postdrome lasts 24 to 48 hours and is a recognized clinical phase. You are not weak. Your brain went through something real.',
                    ]
                },
                {
                    'title': 'IBS taught me more about my own colon than I ever consented to know. Open discussion.',
                    'content': 'I now know the exact transit time of different foods through my gastrointestinal system. I have a mental map of every accessible bathroom within 5km of my office. I can predict bad days with the accuracy of a weather forecast. I have discussed my bowel movements with more people than I ever planned for. This is not a sensitive tummy. This is a genuine chronic condition that has restructured significant parts of my daily life. When are we normalizing digestive health as a serious topic?',
                    'replies': [
                        'The bathroom mental mapping is something every single IBS person does privately and nobody discusses openly. I have rated bathrooms across three cities.',
                        'Low FODMAP with proper structured reintroduction was the most impactful thing I ever did for this condition. The systematic pattern recognition changes everything.',
                        'IBS-C and IBS-D have almost opposite management approaches. Knowing your subtype changes everything about what actually helps.',
                        'The gut-brain axis connection is real. My flares map almost directly onto my anxiety levels. Managing both together is the only approach that worked.',
                        'The sensitive tummy minimization from medical professionals is genuinely harmful. This condition affects employment, relationships, and mental health. It warrants real treatment.',
                    ]
                },
                {
                    'title': 'Chronic pain changes your sex life and nobody in healthcare mentioned this to me once.',
                    'content': 'Got diagnosed with fibromyalgia two years ago. Not one person in my entire care pathway mentioned that chronic pain conditions have significant and well-documented effects on sexual health and intimacy. I found out from a forum. Pain timing, medication effects on libido, relationship strain, all of it. Why is this not a standard part of chronic pain education? For those of you managing this, what actually helped?',
                    'replies': [
                        'Timing is genuinely useful. Most fibromyalgia patients have better physical windows mid-morning when morning stiffness has resolved but before end-of-day fatigue accumulates.',
                        'Pelvic floor physiotherapy made a real difference for me. I was referred by a rheumatologist who actually asked about this. Finding a doctor who asks matters.',
                        'The emotional and relational component is what most affects relationships with chronic illness. Couples therapy specific to chronic illness is underused and genuinely effective.',
                        'My medication which was an antidepressant was affecting my libido significantly. I found this out from an online community not my prescribing doctor.',
                        'You are right that nobody warns you and it is a genuine gap in chronic pain care. The physical and psychological components of intimacy with chronic pain are legitimate medical topics.',
                    ]
                },
            ],
            'Success Stories': [
                {
                    'title': 'Fixed my sleep. Fixed everything else. Furious nobody led with this.',
                    'content': 'Chronic anxiety: significantly better. Low energy: gone. Skin issues: cleared. Mood: more stable. Weight: stopped creeping. I spent years addressing each symptom individually with various interventions and nothing stuck. Then I got serious about consistent sleep, dark cold room, no screens before bed, consistent wake time, magnesium. Eight weeks later my life had changed more than any other single health decision I have ever made. I am annoyed it took this long.',
                    'replies': [
                        'Sleep debt is the silent destroyer of every health metric and nobody talks about it with appropriate seriousness because there is no profitable product to attach to it.',
                        'Which magnesium specifically. Asking on behalf of my entire nervous system.',
                        'Consistent wake time is the actual mechanism. More than duration. Your circadian rhythm is extremely sensitive to that anchor point.',
                        'I had the same skin experience. Spent months on products. Fixed sleep. Skin improved more in six weeks than skincare did in six months.',
                    ]
                },
                {
                    'title': 'Stopped drinking for 90 days as an experiment. Discovered things I did not want to discover.',
                    'content': 'I was not a problem drinker. Two or three drinks four nights a week like every adult I knew. Social. Normal. How adults relax. Stopped for 90 days as a curiosity experiment. Sleep depth improved dramatically within two weeks. Baseline anxiety I had assumed was just my personality dropped noticeably by week four. Lost 4kg without modifying anything else. Realized I had been using alcohol to avoid processing emotions for approximately ten years. That was an uncomfortable thing to notice about myself.',
                    'replies': [
                        'Socially normal drinking in most cultures sits above the level that measurably disrupts sleep quality every single time. The research on this is harsh.',
                        'The anxiety reduction is the finding that always hits people hardest. Alcohol elevates anxiety the next day even at light doses. We have been treating the hangover cure as the disease.',
                        'I did dry January expecting to miss it strongly. I mostly did not. That was more revealing than any physical change.',
                        'The emotion regulation piece is the honest one. Most people are not addicted to alcohol specifically. They are addicted to the relief it provides from feelings.',
                        '4kg in 90 days from removing liquid calories and improving sleep is so real. Hidden calorific content of regular alcohol consumption is radically underestimated.',
                    ]
                },
                {
                    'title': 'At 38 I finally understand my own body and sexual needs. I am annoyed it took this long.',
                    'content': 'Honest post for the women here. In my 20s I was convinced something was wrong with me because my experience never matched what was depicted in media or discussed openly. It took until 35 to actually communicate what I needed to a partner. Until 38 to stop feeling guilty about having those needs. A combination of open communication, actually learning my own responses, and a few sessions with a sexual health therapist genuinely transformed my quality of life in a way no gym programme or supplement ever touched. This is a legitimate health topic.',
                    'replies': [
                        'This is brave to write and I needed to read it today specifically.',
                        'Sexual health education for women specifically is shockingly inadequate. Most of what I know I learned from a book I found by accident, not from any professional or classroom.',
                        'The something is wrong with me feeling is the most universally reported experience when people finally open up about this. Almost nobody had the experience they were shown.',
                        'A licensed sex therapist who is not a sex worker but a clinical professional specializing in sexuality is genuinely underused and evidence-based.',
                        'The communication piece sounds obvious but the vulnerability required is genuinely significant. Starting that conversation is the hardest and simultaneously most valuable step.',
                    ]
                },
                {
                    'title': 'Lost 15kg and also lost the food obsession. The second one is the actual result.',
                    'content': 'People keep asking what my method was and I keep giving the true but boring answer: I stopped dieting. Stopped tracking. Ate when hungry, stopped when full, tried to eat mostly real food without labelling anything forbidden. It took 18 months rather than 6 weeks. There is no app for it. But I no longer think about food constantly, do not feel guilty after meals, and genuinely enjoy eating again for the first time since I was maybe twelve years old. That is the result the scale does not capture.',
                    'replies': [
                        'Intuitive eating gets criticised for lacking structure but what you are describing is the actual intended outcome of that framework.',
                        'The mental freedom from food preoccupation is worth more than any weight loss number. You have described something most diet programmes actively make worse.',
                        'The 18 months is what diet culture hates most because it cannot be packaged or sold. Slow sustainable change is the unfashionable but accurate answer.',
                        'I have been counting calories for eight years. This made me feel something I need to sit with.',
                        'Being able to enjoy eating again is not a minor footnote. Diet culture extracts that joy from people systematically and efficiently. Getting it back is significant.',
                    ]
                },
                {
                    'title': 'Survived full burnout. Here is what the recovery articles do not tell you.',
                    'content': 'Full collapse in February last year. Could not work, could not make basic decisions, cried at emails, forgot words while speaking to people. Took four months of reduced work. Here is what nobody writes about: recovery is not linear. You will have weeks feeling fixed and then crash again. The shame spiral of that crash is worse than the original burnout. The exact skills that made you successful are the same ones that burned you out. And the actual fix is structural change, not a better self-care routine.',
                    'replies': [
                        'The skills that made you successful being the same ones causing the burnout is the part nobody wants to hear and everyone needs to.',
                        'The shame spiral of relapsing is genuinely the hardest phase. You thought you were fixed. You are not fixed yet. You feel like you failed at recovering.',
                        'The real fix is structural should be on the wall of every human resources department and board meeting in existence.',
                        'I tracked my recovery over months. The graph looked like a toddler scribbled on it, not a linear progress chart.',
                        'Burnout is a medical condition. Naming it without caveats and treating it seriously rather than as a productivity problem is the starting point.',
                    ]
                },
            ],
        }

        topics_created = 0
        replies_created = 0

        for category_name, topics_data in data.items():
            try:
                category = ForumCategory.objects.get(name=category_name)
            except ForumCategory.DoesNotExist:
                self.stdout.write(self.style.WARNING(f'Category not found: {category_name}'))
                continue

            for topic_data in topics_data:
                author = random.choice(bot_users)
                topic = ForumTopic.objects.create(
                    category=category,
                    author=author,
                    title=topic_data['title'],
                    content=topic_data['content'],
                    views=random.randint(18, 420),
                )
                topics_created += 1
                self.stdout.write(self.style.SUCCESS(f'+ {topic.title[:70]}'))

                others = [u for u in bot_users if u != author]
                for reply_text in topic_data['replies']:
                    ForumReply.objects.create(
                        topic=topic,
                        author=random.choice(others),
                        content=reply_text,
                    )
                    replies_created += 1

                self.stdout.write(f'  {len(topic_data["replies"])} replies')

        self.stdout.write(self.style.SUCCESS(f'Done! Topics: {topics_created}, Replies: {replies_created}'))
