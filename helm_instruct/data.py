import random
import re

from datasets import Dataset, concatenate_datasets, load_dataset
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.iterable_dataset import IterableDataset

pattern = re.compile(r"\n\nHuman:(.*?)\n\nAssistant:", re.DOTALL)


def _shuffle_and_select(ds, n):
    return ds.shuffle().select(range(n))


def load_self_instruct():
    ds = load_dataset("yizhongw/self_instruct", "human_eval")
    ds = ds["train"]
    ds = _shuffle_and_select(ds, 100)
    ds = ds.map(
        lambda x: {
            "instruction": x["instruction"]
            + " "
            + "\n".join(x["instances"].get("input"))
        }
    )
    ds = ds.rename_column("instruction", "input")

    return ds


def load_koala():
    ds = load_dataset("HuggingFaceH4/Koala-test-set")
    ds = ds["test"]
    ds = _shuffle_and_select(ds, 100)
    ds = ds.rename_column("prompt", "input")
    return ds


def load_vicuna():
    ds = load_dataset("zhengxuanzenwu/vicuna-eval-with-gpt4")
    ds = ds["test"]
    ds = _shuffle_and_select(ds, 80)
    ds = ds.rename_column("instruction", "input")
    return ds


def load_anthropic_red_team_attempts():
    ds: DatasetDict | Dataset | IterableDatasetDict | IterableDataset = load_dataset(
        "Anthropic/hh-rlhf", data_dir="red-team-attempts"
    )
    ds = ds["train"]
    ds = _shuffle_and_select(ds, 100)
    ds = ds.map(
        lambda x: {"transcript": pattern.search(x["transcript"]).group(1).strip()}
    )
    ds = ds.rename_column("transcript", "input")

    return ds


def load_anthropic_harmless_base():
    ds: DatasetDict | Dataset | IterableDatasetDict | IterableDataset = load_dataset(
        "Anthropic/hh-rlhf", data_dir="harmless-base"
    )
    ds = ds["test"]
    ds = _shuffle_and_select(ds, 100)
    ds = ds.map(lambda x: {"chosen": pattern.search(x["chosen"]).group(1).strip()})
    ds = ds.rename_column("chosen", "input")

    return ds


def load_best_chatgpt_prompts():
    language = ["English", "French", "Spanish", "German", "Italian"]
    article = [
        "A ship's wheel or boat's wheel is a device used aboard a water vessel to steer that vessel and control its course. Together with the rest of the steering mechanism, it forms part of the helm. It is connected to a mechanical, electric servo, or hydraulic system which alters the horizontal angle of the vessel's rudder relative to its hull. In some modern ships the wheel is replaced with a simple toggle that remotely controls an electro-mechanical or electro-hydraulic drive for the rudder, with a rudder position indicator presenting feedback to the helmsman."
    ]
    city = ["New York", "Los Angeles", "Toronto", "Berlin", "Paris"]
    event = ["wedding", "funeral", "graduation", "birthday party", "job interview"]
    quote = [
        "Happiness lies in the joy of achievement and the thrill of creative effort. -Franklin D. Roosevelt",
        "Invincibility lies in the defence; the possibility of victory in the attack. -Sun Tze",
    ]
    industry = [
        "machine learning",
        "greentech",
    ]
    description = [
        "a cute puppy",
        "a beautiful sunset",
    ]
    domain_name = ["alphabet-soup.com"]
    keyword = ["allergies", "nutrition"]
    emotion = ["frustration", "joy", "despair"]

    def _get_personal_quote():

        return [
            f"Act as a chef. Write recipes for an {random.choice(language)} three-course meal I can cook for date night.",
            f"Write a casual message in {random.choice(language)} to my Airbnb host saying I’m going to be a little late to check-in and that I will arrive at 4pm.",
            "Write a formal complaint email to United Airlines about my delayed bag from my flight on Tuesday January 17th from New York to Los Angeles.",
            f"Summarize this article into bullet points: {random.choice(article)}",
            "Act as a European travel agent. Come up with a 14-day itinerary for a trip to Germany. The first suggested attraction should be “Take a tour of the Reichstag Building in Berlin.”",
            "Write a letter of resignation to my employer. The reason for my resignation is that I need a more flexible schedule due to family issues.",
            "What’s the best way to make new friends when moving to a new city?",
            "What’s the quickest way to get across Toronto during rush hour?",
            f"Translate the following text into {random.choice(language)}: {random.choice(article)}",
            f"List 5 of the best bars in {random.choice(city)}.",
            f"Act as a tailor. Pick an appropriate outfit for a {random.choice(event)}.",
            f"Respond to this text message below from my mom {random.choice(article)}",
        ]

    def _get_funny_quote():
        return [
            "What’s the best prank to play on a friend?",
            "Send a pun-filled happy birthday message to my friend Alex.",
            "Make a joke about chickens.",
            "Write a parody song about the alphabet.",
            "What do you get when you cross a snowman and a vampire?",
            "Write a short story where a pencil is the main character.",
            "What would happen if dogs could talk?",
            "Write a fictional news headline about robots taking over the world.",
            "Create a silly dialogue between two inanimate objects.",
            "What would happen if the moon were made of cheese?",
        ]

    def _get_student_quote():
        return [
            "Act as a college interviewer for a Business School. Help me come up with questions I should ask the interviewer at the end of the interview.",
            "Act as a tutor. I need help understanding how the quadratic formula works. Please describe it in easy-to-understand terms.",
            "How can I modify the Pomodoro technique to suit my method of study?",
            f"Explain possible meanings of this quote: {random.choice(quote)}",
            "Can you devise practical ways to stay focused during long study sessions?",
            "Help me find a way to balance my studying and social life.",
            "Structure a 1,500-word essay on Max Planck’s quantum theory.",
            "Come up with 10 ways to improve memory and recall while studying for exams.",
            "List note-taking techniques for a chemistry lecture.",
            "Suggest 10 Chrome extensions for students designed to improve productivity while studying.",
        ]

    def _get_marketing_quote():
        return [
            "Write a personalized blog post promoting my latest WordPress theme bundle.",
            "How do I increase my Twitter followers?",
            "Generate content ideas for my SaaS company.",
            "Produce 50 hashtags",
            "Create a TikTok campaign plan for launching an exciting new low carb mac and cheese, aimed at Gen Z and millennial consumers.",
            "Suggest inexpensive ways I can promote my plumping business without using social media.",
            "Is investing in influencer marketers worth the cost?",
            "How can I grow our brand’s TikTok audience?",
            "How can I use YouTube to increase brand awareness?",
            "Write a product description for my latest set of landscape oil paintings of the Scottish Highlands.",
            "Generate high-ticket offerings for my online language course.",
            "How can I use TikTok to increase sales conversions?",
            "Write a minute-long script for an advertisement about new sneakers.",
        ]

    def _get_midjourney_quote():
        return [
            f"Write a good prompt for an image generation AI to make an image of this: {random.choice(description)}",
            "Generate a detailed description of an AI-generated cityscape with a futuristic twist.",
            "Create an image description that describes a visually stunning setting that takes place in the year 3030.",
            "Design with words an abstract composition with a graphic, minimalist style.",
            "With distinct adjectives, create a visual with words that would encompass the feeling of being lost in life",
        ]

    def _get_entrepeneur_quote():
        return [
            f"Analyze the current state of {random.choice(industry)} and its trends, challenges, and opportunities, including relevant data and statistics. Provide a list of key players and a short and long-term industry forecast, and explain any potential impact of current events or future developments.",
            "Provide a step-by-step guide on creating a business plan, including key components, useful resources, and tips for success.",
            "Write a comprehensive and easy-to-understand explanation of different marketing strategies and their effectiveness for small businesses.",
            "Offer a comprehensive guide to small business financing options, including loans, grants, and equity financing.",
            "Provide a guide on managing finances for a small business, including budgeting, cash flow management, and tax considerations.",
            "Write an in-depth analysis of the current state of a specific industry and its potential for small business opportunities.",
            "Offer a detailed review of a specific software or tool for small business operations, such as accounting, project management, or CRM.",
            "Write a detailed explanation of the pros and cons of outsourcing vs in-house for small business operations.",
            "Offer an in-depth analysis of the current state of small business legislation and regulations and their impact on entrepreneurship.",
            "Provide a guide on networking and building partnerships as a small business owner.",
            "Present a list of valuable resources and organizations for small business support and growth.",
        ]

    def _get_blogging_quote():

        return [
            "Write a brief for a blog post about opening a Gumroad store.",
            "Generate 5 social media posts for my blog post on AppSumo.",
            "Pick five keywords for a blog post titled “10 ways to improve my photography skills.”",
            "Suggest engaging titles for a blog post about 1930s Art Deco architecture.",
            f"Generate user-friendly URLs for the domain {random.choice(domain_name)} for these keywords: {random.choice(keyword)}.",
            "Create a content calendar with six blog titles, including the keyword {random.choice(keyword)}. Pick suitable publishing dates for each guide spread across May 2023.",
            f"Write a creative outreach email for a guest post pitch for the keyword {random.choice(keyword)} for the domain {random.choice(domain_name)}. Come up with 3 title ideas using the keyword.",
        ]

    def _get_creative_quote():
        return [
            "Write a scary short story about a man trapped in an abandoned house.",
            "Generate five synonyms for sublime.",
            "Write a backstory for a 55-year-old male character during the French Revolution.",
            "Write the first stanza of a poem about cabbages with an AABB rhyme scheme.",
            "Write hilarious fan fiction about the Twilight saga.",
            "How should I pace a science fiction novella about traveling to Saturn’s moon, Titan?",
            "Act as an 18th-century pirate. Describe what life was like on a pirate ship in Southeast Asia.",
            "Write the opening to a story from the point of view of a washing machine.",
            "How can I make a soliloquy engaging at the beginning of a play?",
            "Describe ways I can make a framed narrator relevant to a story.",
            "Continue this dialogue between a store clerk and a police officer:\nA: Did you see anything suspicious yesterday afternoon?\nB: Yesterday was...Sunday, I don't remember anything out of the ordinary.",
        ]

    def _get_copywriting_quote():
        return [
            "Act as a copywriter. Write long-form copy for the Hard Rock Cafe in Macau promoting merchandise.",
            "Act as a copywriter. Write short-form copy for a billboard in Times Square promoting Wicked the Musical.",
            "How is short-form copywriting easier than long-form copywriting?",
            "How is copywriting different from SEO content writing?",
            "How does repetition improve short-form copywriting?",
            "What is the PAS formula? And give 3 examples of the PAS formula being used.",
            "Provide examples of successful copywriting campaigns that use repetition",
            "Give examples of newspaper headlines that grab the reader’s attention.",
            "How can I integrate copywriting into social media posts?",
            "List unusual copywriting techniques that I can use to create taglines.",
        ]

    def _get_health_quote():
        return [
            "List the top 10 healthy foods to include in my diet.",
            "Develop a 30-day workout routine to help me lose 2 lbs a week.",
            "Act as a nutritionist. Help me devise 10 healthy meals that can be cooked in 30 minutes or less.",
            "Provide a guide on healthy nutrition for weight management and weight loss.",
            "Create a 1 month workout plan for me exercise my shoulder muscles.",
            "Explain the benefits of daily exercise and provide a sample workout plan.",
            "Write a comprehensive guide on managing stress and maintaining mental wellness.",
            "Provide a list of common sleep disorders and tips for improving sleep quality.",
            "Explain the different types of therapy and their effectiveness in treating mental health issues.",
            "Offer a detailed explanation of the benefits and risks of alternative medicine practices, such as acupuncture and herbal remedies.",
            "Write an in-depth analysis of the current state of the healthcare system and its impact on the general population.",
            "Offer a list of recommended resources for quitting smoking and managing addiction.",
            "Explain the importance of regular medical check-ups and preventive care.",
        ]

    def _get_event_quote():
        return [
            "Create a checklist for event planning, including important tasks and deadlines.",
            "What are some creative ways to add personal touches to the seating at a wedding dinner?",
            f"Provide a list of top event venues in {random.choice(city)}, along with their capacities.",
            "Write a guide on event budgeting, including tips for saving money and avoiding common overspending pitfalls.",
            "List the best catering companies in a specific area, along with menu options and prices.",
            "Offer a comprehensive explanation of event marketing, including target audience analysis and promotion strategies.",
            "Explain the importance of wedding photography, including styles, techniques, and essential shots to capture.",
            "Explain the different types of event equipment rental options, including audiovisual, lighting, and decor.",
            "Write a guide on event security, including necessary measures for crowd control and emergency response planning.",
            "Provide a list of event management software and tools, along with their key features and benefits.",
            "Offer an in-depth analysis of current event industry trends, including popular themes, formats, and technologies.",
            "Write a guide on event evaluation, including metrics for measuring success and feedback mechanisms for continuous improvement.",
        ]

    def _get_designer_quote():
        return [
            f"Create a mood board for a design project that evokes {random.choice(emotion)}.",
            "What are the best online marketplaces for selling designs?",
            "How can I create a minimalistic logo that conveys a strong brand image?",
            "What design elements should I consider when creating a packaging design for a luxury brand?",
            "How can I create an eye-catching poster design for an upcoming event?",
            "What color palette would be appropriate for a law firm’s website design?",
            "How can I design a user-friendly interface for a mobile application?",
            "What font and typography techniques should I use to create a professional-looking business card?",
            "How can I create an animated graphic that effectively communicates a complex idea?",
            "What design elements should I include in a brochure to promote a real estate development?",
        ]

    def _get_artist_quote():
        return [
            "Which factors determine the price of my artwork?"
            "How can I develop my own unique style as an artist?",
            "What techniques can I use to create a captivating digital illustration?",
            "How can I create a compelling concept for a series of illustrations?",
            "What tools and materials should I use for traditional watercolor painting?",
            "How can I create a realistic portrait in pencil or charcoal?",
            "What methods can I use to incorporate text into my illustrations?",
            "How can I create an engaging comic strip or graphic novel?",
            "What steps should I take to prepare a portfolio for job applications or exhibitions?",
            "How can I develop a successful freelance illustration business?",
            "What resources and communities are available to artists and illustrators for inspiration and professional development?",
        ]

    def _get_web_quote():
        return [
            "Act as a software engineer. Come up with an architecture and code for developing a random winner picker website with JavaScript.",
            "Please continue writing this code for JavaScript:\nfunction getTotalSum(numbers) {",
            "Provide a UX design tip I can share on Instagram.",
            "Help me find mistakes in the following code:\nfor (const i = 0; i <= array.length; i+) {\n  sum += array[i];\n}",
            "List ways I can use AI in software engineering.",
            "What are 5 of the best practices for software architecture design?",
            "What are the tips and tricks for writing efficient code?",
            "Suggest tools I can use to make writing code easier.",
            "How do I make an accessible Tailwind Footer?",
            "Write a docstring for the following function:\ndef compute(items):\n\treturn sum(x**2 for x in items if x is not None)",
            "I’m making a website for a small business that sells hand-crafted furniture. I need ideas on how to structure the website using WordPress.",
        ]

    def _get_project_quote():
        return [
            "Create a workback schedule for a remodeling project, with a timeline of 6 months, with the deadline of August 1.",
            "Act like a project manager and create a high-level project plan for a new product launch.",
            "How can I effectively communicate my current web development project’s progress and status to stakeholders?",
            "How can I effectively prioritize tasks and allocate resources in a complex digital advertisement campaign project?",
            "What tools and methodologies can I use to manage my project’s risk?",
            "How can I motivate and engage a remote or virtual team?",
            "What strategies can I use to effectively manage project scope and budget?",
            "What approaches can I use to effectively manage and resolve conflicts within a project team?",
            "What processes should I put in place for continuous improvement and project optimization?",
            "How can I create a project schedule that accurately reflects task dependencies and resource constraints?",
            "What techniques can I use to successfully manage multiple projects simultaneously?",
        ]

    def _get_seo_quote():
        return [
            "Write a 100-character meta description for my blog post about classical piano.",
            "Come up with 5 long-tail keywords for a post about how to create a DIY slat wall.",
            "What are 5 ways I can improve SEO on my food blog?",
            "Write a casual backlink outreach email to Alice to tell them about why they should consider switching a link out on their 'Best Bay Area hiking trails' post with my resource.",
        ]

    def _get_email_quote():
        return [
            "Come up with 5 short email subject lines for our brand’s new launch of a lavender soap line, include an emoji at the beginning.",
            "Write follow-up email for people who attended my precious metals webinar.",
            "Generate subject line for a Black Friday sale email.",
            "Structure a weekly fitness newsletter.",
            "Write body copy for my vegan restaurant’s new menu launch.",
            "Create a personalized email greeting for a VIP customer.",
            "Create 5 ideas for an email campaign promoting eco-friendly products.",
            "Help me boost open rates with a compelling email subject line for a book club.",
            "Create 5 compelling CTAs to prompt donations for a charity fundraising marathon.",
            "How do I ensure my marketing emails look good on iOS and Android?",
            "How can I increase the click-through rate on my marketing emails?",
            "What is A/B testing and how can it improve email engagement?",
        ]

    def _get_social_quote():
        return [
            "Generate 5 hashtags for a new Instagram post about our latest product launch.",
            "Create a captivating tweet to announce our new partnership.",
            "Come up with a list of 5 influencer outreach messages for a product collaboration.",
            "Generate a 2-minute video script for a Facebook ad campaign promoting our new service.",
            "Create a 1-paragraph blog post about the benefits of using our new app for social media management.",
            "Come up with 10 creative Instagram story ideas for a beauty brand.",
            "Generate a creative social media content calendar for the next month.",
            "Create a series of 5 Instagram posts to showcase our brand values.",
            "Write a catchy Instagram bio for a new food delivery service.",
            "Generate a series of 5 Twitter polls for market research on our target audience.",
            "Come up with a list of 10 engaging Facebook post ideas for a fitness brand.",
            "Create a LinkedIn post to announce a job opening in our company.",
            "Generate 5 creative ways to use Instagram Reels for a fashion brand.",
            "Write a persuasive tweet to promote a new book.",
            "Come up with a series of 5 Instagram posts to showcase customer success stories.",
            "Generate a list of 10 questions for a Q&A session on Instagram Live.",
            "Create a catchy TikTok hashtag challenge for a new product launch.",
            "Write a Twitter thread to explain the features of a new app.",
            "Come up with a list of 5 Pinterest boards to showcase our brand’s products.",
            "Generate a series of Facebook ads to promote an upcoming sale.",
        ]

    def _get_content_quote():
        return [
            "What type of camera should I consider for daily vlogging?",
            "What are some creative ways to grow my Twitch audience?",
            "Write an outline for a YouTube video script for an iPhone 14 Pro Max review.",
            "What factors should I consider when quoting for a brand deal with a candle company, and what ballpark range should I charge? The scope is to post 3 videos on TikTok, and I have 100,000 followers.",
            "Come up with 5 catchy Instagram caption ideas for my latest vlog on hiking in Switzerland.",
            "Generate a script for a 60-second Instagram Reel for a Gen Z fashion brand.",
            "Come up with a list of 10 attention-grabbing headlines for a food influencer.",
            "Generate a persuasive email to a potential sponsor for a YouTube channel.",
            "Write a list of 5 topics to cover in a podcast episode for a personal finance show.",
            "Come up with a list of 10 Instagram post captions for a fitness influencer.",
            "Generate a script for a 2-minute Instagram story for a beauty brand.",
            "Write a list of 5 YouTube video ideas for a gaming channel.",
            "Come up with a list of 10 Twitter threads to start for a political commentator.",
            "Generate a list of 5 Pinterest boards to create for a home décor influencer.",
            "Come up with a list of 10 hashtags to use for a nature photographer’s Instagram posts.",
            "Generate a script for a 30-second commercial for a local business.",
            "Write a list of 5 topics to cover in a video for a cooking channel.",
            "Come up with a list of 10 Facebook post ideas for a pet store.",
            "Generate a list of 5 LinkedIn articles to write for a business consultant.",
            "Write a list of 5 TikTok video ideas for a dance influencer.",
            "Come up with a list of 10 Pinterest pins to create for a wedding planner.",
        ]

    def _get_sales_quote():
        return [
            "Write a cold email to a prospective customer to introduce them to my llama walking company and how it can benefit them.",
            "Create a personalized sales email for a potential customer for my dancing robot company.",
            "Qualify this lead based on their behavior and interests.",
            "Segment our customers based on their buying behavior.",
            "Provide chat-based support for customer inquiries about our product.",
            "What complementary products would you recommend for this customer?",
            "What are some creative ways to generate leads for my dancing robot company?",
            "Provide after-sales support and upselling opportunities for my llama walking product.",
            "What cross-selling opportunities would you recommend for my dancing robot business?",
        ]

    def _get_real_quote():
        return [
            "Generate a list of 10 prospective client follow-up messages.",
            "Write a compelling property listing for a spacious 3-bedroom, 2-bathroom loft in SoHo, Manhattan.",
            "Write a persuasive email to a potential home seller.",
            "Create a 2-minute virtual tour script for a property listing.",
            "Create a list of 5 local hotspots to mention in a neighborhood guide.",
            "Write a 1-page property brochure for a new listing.",
            "Write a captivating property description for an online listing.",
            "Come up with a series of 5 social media posts to showcase your listings.",
            "Generate a list of 10 home-buying tips for first-time buyers.",
            "Write a persuasive letter to a property owner about listing their property with you.",
            "Come up with a list of 5 home staging tips for sellers.",
            "Generate a list of 10 potential clients from your network.",
            "Write a persuasive message to a client who is relocating to a new city.",
            "Come up with a series of 5 open house ideas.",
            "Generate a list of 10 potential leads from expired listings.",
            "Write a follow-up email to a client who recently viewed a property.",
            "Come up with a list of 5 reasons to choose your real estate company.",
            "Generate a persuasive message to a client considering renting instead of buying.",
            "Write a series of 5 Facebook ads to promote a new housing development.",
            "Come up with a list of 10 local real estate market trends to discuss with clients.",
            "Generate a list of 5 home-buying pitfalls to warn clients about.",
            "What are the benefits of working with a real estate agent when buying or selling a property?",
            "How do you determine the market value of a property?",
            "What is the home buying process like and how can I prepare for it?",
            "What are some common mistakes that buyers make when purchasing a home?",
            "How can I stage my home to appeal to potential buyers?",
            "What is the current real estate market trend in [insert location]?",
            "How can I negotiate the best price for my home?",
            "What documents do I need to have in order before buying or selling a property?",
            "What are the latest technology and marketing tools used to promote properties?",
            "Explain the different financing options available to home buyers",
        ]

    def _get_resume_quote():
        return [
            "Write a cover letter for a software engineer position highlighting my technical skills.",
            "Generate a personalized objective statement for a marketing resume.",
            "Come up with a list of 5 relevant achievements to include in a financial analyst cover letter.",
            "Generate a tailored 2-minute pitch for a sales job interview.",
            "Write a persuasive email to a potential employer explaining my background as a nurse.",
            "Come up with a list of 10 unique qualities to include in a teacher’s resume.",
            "Generate a 1-page summary of my experiences and accomplishments as a graphic designer.",
            "Write a cover letter addressing the specific qualifications listed for a project manager position.",
            "Come up with a list of 5 ways to tailor my resume for a customer service job.",
            "Generate a list of 10 keywords to include in a human resources resume and cover letter.",
            "Write a persuasive letter to a hiring manager explaining a gap in my work history as a lawyer.",
            "Come up with a list of 5 quantifiable results to highlight in a business analyst resume.",
            "Generate a list of 10 relevant skills and experiences for a web developer job application.",
            "Write a personalized thank you note to a potential employer after a doctor job interview.",
            "Come up with a list of 5 personal traits that make you a strong fit for a social worker role.",
            "Generate a 2-minute response to common interview questions for a data scientist position.",
            "Write a persuasive email to a potential employer negotiating a higher salary for a software developer role.",
            "Come up with a list of 10 professional references for an administrative assistant job application.",
            "Generate a list of 5 ways to make my resume stand out from other applicants for a journalist position.",
            "Write a persuasive message to a potential employer explaining my relocation for a chef role.",
        ]

    def _get_product_quote():
        return [
            "Outline 5 potential features to enhance a food delivery app.",
            "Compile a market analysis report for a cutting-edge smartwatch.",
            "Identify 10 potential partnership opportunities for a ride-sharing company.",
            "Create a user flow diagram for a mobile app connecting users to local volunteer opportunities.",
            "Propose 5 solutions to improve the user experience on an e-commerce website.",
            "Prepare a competitor analysis report for a revolutionary virtual reality headset.",
            "Devise 10 possible integrations for a smart home automation system.",
            "Draft a product requirements document for a new and improved video conferencing tool.",
            "Suggest 5 ways to streamline the checkout process on an online store.",
            "Develop user personas for a new tablet designed to educate kids.",
            "Uncover 10 potential upsell opportunities for a successful meal kit subscription service.",
            "Build a product roadmap for a state-of-the-art fitness app.",
            "Propose 5 ways to simplify the onboarding process for a project management tool.",
            "Map out a customer journey for a novel pet care delivery service.",
            "Explore 10 potential collaborations for a green electric scooter rental company.",
        ]

    ds = Dataset.from_dict(
        {
            "input": _get_personal_quote()
            + _get_funny_quote()
            + _get_student_quote()
            + _get_marketing_quote()
            + _get_midjourney_quote()
            + _get_entrepeneur_quote()
            + _get_blogging_quote()
            + _get_creative_quote()
            + _get_copywriting_quote()
            + _get_health_quote()
            + _get_event_quote()
            + _get_designer_quote()
            + _get_artist_quote()
            + _get_web_quote()
            + _get_project_quote()
            + _get_seo_quote()
            + _get_email_quote()
            + _get_social_quote()
            + _get_content_quote()
            + _get_sales_quote()
            + _get_real_quote()
            + _get_resume_quote()
            + _get_product_quote()
        }
    )
    ds = _shuffle_and_select(ds, 100)
    return ds


def load_oasst1():
    ds = load_dataset("OpenAssistant/oasst1")
    ds = ds["validation"]
    ds = ds.filter(lambda x: x["lang"] == "en")
    ds = _shuffle_and_select(ds, 100)
    ds = ds.rename_column("text", "input")
    return ds


def load_data_helm_insruct():
    ds = concatenate_datasets(
        [
            load_self_instruct().select_columns(["input"]),
            load_koala().select_columns(["input"]),
            load_vicuna().select_columns(["input"]),
            load_anthropic_red_team_attempts().select_columns(["input"]),
            load_anthropic_harmless_base().select_columns(["input"]),
            load_best_chatgpt_prompts().select_columns(["input"]),
            load_oasst1().select_columns(["input"]),
        ]
    )
    return ds
