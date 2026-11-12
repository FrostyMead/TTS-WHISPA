"""Short paragraphs for the voice-training reading session (original text, MIT licensed).

Each one is about 12-18 seconds of reading at a normal pace (25-40 words): several times
the speech of a single sentence for the same press, check and save, and still under the
20-second limit the fine-tuning uses for one clip.
"""

PASSAGES: tuple[str, ...] = (
    # --- everyday messages
    "Hey, just a heads up that I'm running about ten minutes late. The traffic on the bridge "
    "is terrible this morning, so go ahead and order a coffee for me and I'll pay you back.",
    "Thanks again for dinner last night. The lamb was cooked perfectly and the dessert was even "
    "better. Next time it's my turn to host, so let me know which weekend suits you.",
    "Could you grab some bread, milk and a dozen eggs on your way home? We're also out of "
    "dishwashing liquid, and the dog needs another bag of food before the weekend.",
    "I left the spare key under the pot plant by the back door. The cat has already been fed, "
    "but she'll probably pretend she hasn't, so please don't believe her.",
    "Sorry I missed your call earlier. I was stuck in a meeting that ran way over time. I'm "
    "free after four this afternoon if you'd still like to have a chat.",
    "Happy birthday! I hope you have a wonderful day with your family. We should catch up for "
    "lunch next week and celebrate properly, my shout this time.",
    "The plumber is coming tomorrow morning sometime between eight and twelve. If I'm not home "
    "yet, can you let him in and show him the leak under the kitchen sink?",
    "We just got back from the beach and everyone is exhausted. The kids built an enormous "
    "sandcastle, and then a wave washed the whole thing away in about five seconds.",
    "I've booked a table for six at the Thai place near the station. It's at seven thirty on "
    "Friday, so let me know if anyone else wants to come along.",
    "My phone is about to die, so I might not reply for a while. If anything urgent comes up, "
    "just send me an email and I'll check it when I get to the office.",
    "Just landed and waiting for my bag. The flight was smooth, but the person next to me "
    "snored the entire way. I'll call you once I'm in the taxi.",
    "Thank you so much for helping me move last weekend. I honestly couldn't have done it "
    "without you, and I still owe you a couple of beers for carrying the couch.",
    "Can you remind me to call the bank about my card before lunch? I keep forgetting, and "
    "they said the new one would arrive within five business days.",
    "The weather looks great for Saturday, so the barbecue is definitely still on. I'll bring "
    "the sausages and a salad, and maybe you could bring some drinks and ice?",
    "I finally finished that book you lent me. The ending completely surprised me, and I "
    "stayed up far too late reading the last few chapters in one go.",
    "We're thinking of painting the living room before the holidays. Something warm and "
    "light, maybe a soft grey or a pale green. What do you think would work?",
    "The package arrived this afternoon, but the box was a bit damaged on one corner. "
    "Luckily everything inside looks fine, and the colours are exactly what I ordered.",
    "I'm heading out for a run along the river and should be back in about an hour. There's "
    "leftover pasta in the fridge if you get hungry before I'm home.",
    "It was lovely catching up yesterday. I didn't realise how much I'd missed our long chats. "
    "Let's not leave it another six months before we do it again.",
    "Our flight has been delayed by two hours, so we won't make it to dinner tonight. Please "
    "start without us, and save us a slice of that famous cheesecake.",
    # --- work and meetings
    "Hi everyone, thanks for joining. Today we'll go through the project timeline, review the "
    "budget, and agree on the next steps. Please hold questions until the end.",
    "Just a quick update on the launch. Testing is finished, the documentation is nearly "
    "done, and we're still on track for the release at the end of the month.",
    "Could you send me the latest version of the proposal before Thursday? I'd like to read "
    "it properly and add a few comments before we present it to the client.",
    "I've moved our weekly catch up to Wednesday afternoon, because half the team is away on "
    "Monday. The agenda is the same, so there's nothing extra to prepare.",
    "Following on from yesterday's meeting, I've summarised the main decisions and the action "
    "items. Please check that your name is next to the right tasks and reply by Friday.",
    "The client was really happy with the demo. They asked about pricing for a larger team "
    "and want a follow up call next week to talk through the contract.",
    "I'll be working from home tomorrow, because the heating in our building is broken "
    "again. I'll still join every meeting by video, so nothing needs to change.",
    "We need to hire two more people before the end of the quarter. I've drafted the job "
    "descriptions, and I'd love your feedback on the required experience section.",
    "Thanks for the great work on the report. The charts are much clearer now, and the summary "
    "at the top makes it easy for the board to see what matters.",
    "Let's keep this meeting to thirty minutes. We only need to decide on the venue, the date, "
    "and who is responsible for sending out the invitations.",
    "The budget is tighter than we expected, so we'll have to choose between the new "
    "website and the marketing campaign. I'd suggest we finish the website first.",
    "Please remember to submit your timesheets by five o'clock today. Payroll runs tomorrow "
    "morning, and late timesheets will have to wait until the next pay cycle.",
    "I've shared a folder with all the training material. Start with the overview, then work "
    "through the exercises in order. It should take about two hours in total.",
    "Quick reminder that the office is closed on Monday for the public holiday. Enjoy the long "
    "weekend, and we'll pick everything up again on Tuesday morning.",
    "The supplier says the parts will be two weeks late. I've asked whether they can send a "
    "partial order now, so the team can at least start the first stage.",
    "Great job, everyone. The event went smoothly, the feedback has been fantastic, and we "
    "came in slightly under budget. Drinks are on me on Friday afternoon.",
    # --- tech and software
    "I pushed a fix for the login bug to a new branch. Could you review the pull request when "
    "you have a moment? The tests are passing, but I'd like a second opinion.",
    "The server ran out of disk space overnight, which is why the website went down. I've "
    "cleared the old log files and added an alert so it won't happen again.",
    "Before you deploy, make sure the database migration has run on the staging environment. "
    "Last time we skipped that step and spent the whole afternoon rolling back.",
    "The app crashes when you rotate the screen on older phones. I think it's a memory issue "
    "in the image viewer, so I'm going to profile it this afternoon.",
    "Let's switch the build to the new version of the compiler. It's faster, it gives "
    "clearer error messages, and it fixes the warning we keep seeing in the logs.",
    "I've written a short script that backs up the shared folder every night and deletes "
    "copies older than thirty days. It runs at two in the morning, when nobody is working.",
    "The API returns an error when the request is missing the date field. We should validate "
    "the input properly and send back a helpful message instead of a crash.",
    "Can you restart the router? The Wi-Fi keeps dropping out in the meeting room, and the "
    "video call froze three times during this morning's presentation.",
    "I updated the laptop overnight, and now the printer doesn't show up anymore. I've "
    "reinstalled the driver twice, so I think it might be a network setting.",
    "The new dashboard loads in under a second now. We cached the slow queries and removed a "
    "chart that nobody was using, which made a surprisingly big difference.",
    "Please don't share passwords in the chat. Use the password manager instead, and turn on "
    "two factor authentication for every account that supports it.",
    "The spreadsheet has a formula error in the totals column. Somebody deleted a row, so "
    "everything below it is now pointing at the wrong cells.",
    # --- numbers, dates and money
    "The meeting starts at 3:30 PM on Thursday, so please be in the room by 3:25. We have a "
    "lot to cover and only forty five minutes before the next booking.",
    "Our sales grew by 18% in the third quarter, mostly thanks to the online store. If the "
    "trend continues, we'll pass last year's total by the end of November.",
    "The new laptop costs $1,899, but there's a 10% discount for staff until the end of the "
    "month. That brings it down to just under $1,710.",
    "The conference runs from March 12 to March 14. Registration opens at 8 AM each day, "
    "and the keynote begins at 9:15 in the main hall.",
    "We need 120 chairs, 15 tables and enough food for about 150 people. The catering company "
    "needs final numbers at least one week before the event.",
    "Our team grew from 12 people to 45 people in just under two years. We'll need a bigger "
    "office soon, ideally somewhere close to the train line.",
    "The rent is going up by $40 a week from the first of July. That's still reasonable for "
    "the area, but we should review our budget before signing the new lease.",
    "Please upgrade to version 2.4 before Friday. The older version stops receiving security "
    "updates next month, and some features won't work after that.",
    # --- people and places
    "My cousin Sarah is flying in from Melbourne next Tuesday, and she's staying with us for "
    "a week. I'm planning to take her up to the Blue Mountains on the weekend.",
    "We spent three days in Tokyo and then caught the train down to Kyoto. The temples were "
    "beautiful, and the food was better than anything I've had in years.",
    "Doctor Patel said the results look normal, but she wants to see me again in six months, "
    "just to be safe. The appointment is booked for early June.",
    "I met James and Priya at the airport, and we drove straight to the coast. The road "
    "through the hills was slow, but the views made it completely worth it.",
    "Our neighbours, Tom and Mei, invited us over for dinner on Saturday. They've just "
    "renovated their kitchen, and apparently the new oven is enormous.",
    "The train from Brisbane to the Gold Coast takes about an hour and a half. If we leave "
    "early enough, we'll have the whole afternoon at the beach.",
    "Please forward this to Michael in finance and copy in Olivia from legal. They'll both "
    "need to approve the contract before we can send it to the supplier.",
    "We're driving to Adelaide for my grandmother's ninetieth birthday. The whole family will "
    "be there, including a few cousins I haven't seen since I was a kid.",
    # --- questions and conversation
    "Do you have a minute to go over the plan with me? I'm not sure I understand the second "
    "part, and I'd rather ask now than get it wrong later.",
    "What time works best for you next week? I'm free most mornings except Wednesday, and I "
    "can do a quick call in the afternoon if that's easier.",
    "Have you tried the new cafe on the corner? Everyone keeps telling me the coffee is "
    "amazing, but there's always a long queue whenever I walk past.",
    "Would it be possible to push the deadline back by a few days? The data arrived late, "
    "and I want to make sure the analysis is done properly.",
    "How did the interview go? I hope they asked about your design work, because that "
    "portfolio is honestly one of the best I've seen in a long time.",
    "Is anyone using the meeting room after lunch? If not, I'd like to book it for an hour to "
    "run a quick training session for the new starters.",
    "Which option do you prefer, the cheaper one with slower delivery or the more expensive "
    "one that arrives tomorrow? I'm happy with either, so it's your call.",
    "Did you get my message about the weekend? We're still keen to go camping, but we need to "
    "know by Thursday so we can book the site before it fills up.",
    # --- describing and explaining
    "To reset the device, hold the power button for ten seconds until the light turns orange. "
    "Then release it, wait a moment, and press it once more to start again.",
    "The recipe is really simple. Fry the onion and garlic, add the tomatoes and a pinch of "
    "salt, then let it simmer for twenty minutes while the pasta cooks.",
    "The view from the top of the hill was incredible. You could see the whole city, the "
    "harbour, and the mountains in the distance, all glowing in the late afternoon light.",
    "First, open the settings menu and choose accounts. Next, select the account you want to "
    "change, enter your new email address, and confirm it using the code we send you.",
    "The museum was much bigger than I expected. We spent most of the day in the science "
    "section, and the kids loved the room full of dinosaur skeletons.",
    "The storm knocked out the power for about six hours. We lit some candles, played cards, "
    "and ended up having one of the nicest evenings we've had in ages.",
    "Learning to play the guitar has been harder than I thought. My fingers hurt after every "
    "practice, but I can finally play a few songs without stopping.",
    "Our garden is finally starting to come together. The tomatoes are nearly ready, the "
    "herbs are growing like crazy, and the lemon tree has its first flowers.",
    # --- longer, punctuation-heavy
    "Honestly, I'm not sure it's the right choice. It's cheaper, sure, and it's available "
    "right now, but it doesn't do half of what we need. Let's keep looking.",
    "\"Are you sure this is the right way?\" she asked, looking at the map. I wasn't, of "
    "course, but I nodded anyway and kept driving towards the hills.",
    "Good news: the package arrived, the colours are right, and everything fits perfectly! "
    "The only problem is the instructions, which seem to be written for a different model.",
    "It's not that I don't like the idea. It's just that we've tried something similar "
    "before, and it didn't work, mostly because nobody had time to keep it going.",
)
