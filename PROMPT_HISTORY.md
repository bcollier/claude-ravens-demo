# Every instruction given, in order

The complete set of prompts behind this project, quoted exactly as typed. Extracted from the Claude Code session transcript by `scripts/prompt_history.py`, not written from memory.

It is worth reading as its own artefact. Almost none of these prompts specify *how* to do anything. They state a goal, and several of the most consequential ones are corrections or challenges to work already done — asking for a neural network that had been left out, asking whether the evaluation was honest, asking what had actually been achieved by a particular minute. The quality of the result owes more to those than to the opening brief.

All times are local, on the day of the session. 18 messages.

---

## 1. 12:58

> all right, Claude. I want to build a tool that will solve the Raven's intelligence tests. I'm going to give you a repo, and you can pull that down, and I want you to do three things:
>
> 1. I want you to get the code that's in the repo to work. The code is eight years old, I don't even know if it works anymore. I want you to show how good that code is, how many it gets right in one application.
>
> 2. I want you to build another application that will use any kind of technique except for an LLM. You're not allowed to use an LLM, but you can use any AI technique you want. I want you to go through all 96 of these puzzles and solve them using any kind of ML technique or AI technique that you want.
>
> 3. I want a third program that is going to use an LLM. I will give you an OpenAI key, so you can use the OpenAI libraries. Assume you can use GPT-5.6 or whatever you think is the most intelligent model right now to solve these.
>
> At the end, I'm going to want a comparison of my original code, the code with AI or ML that isn't an LLM, and then an LLM as well. You have my GitHub key, so you can push all of these to new repos. Actually, just make one repo called "Claude Raven's Demo" and push all of that to it.
>
> I'd like a markdown file comparing the performance across all three. If you get stuck, let me know, but I'm going to be teaching, so I don't want you to really ask me a bunch of questions. I'll paste the link right now.  https://github.com/bcollier/KBAI_Ravens_Project   ANy questions before you start?

## 2. 13:58

> yes go ahead, also make sure you record my full original prompt, what model you are currently running etc.

## 3. 14:03

> if possible I would also like to know how many tokens you spent on this task, including input, output, cache, reasoning, etc.  I want to know how many lines of code you generated, things like that. Give me visual representations of the architecture of the code, the logic behind why you chose the approach, what you tried along the way, the whole narrative. Everything that would be helpful for students learning about how AI works

## 4. 14:05

> also explain the original code, what was it doing, what was effective about what it was doing and what was lacking

## 5. 14:07

> also I want to know what you had accomplished by exactly 1:50PM today. look through the logs to tell me the answer to the following questions students were asked:
>
> Will Claude Code be able to solve any (1 or more) Raven's Matrices problem without the use of an LLM by the end of class?
>
> How many out of 96 do you estimate the non-LLM version of this program will be able to solve by the end of class?

## 6. 14:24

> add a new section called something like "comparisons added after class" 
>
> (1) for some reason you didn't use a neural network or thought I didn't want a neural network. build a model that is as good as possible with any method except for LLM's including a neural network
>
> (2) this is all testing on data that it trained on correct? split into a train/test split of 70/30 and rerun the analysis for any method that used machine learning. do not touch the original work, add this onto the comparison as an Appendix or epilogue.
>
> (3) compare the performance of GPT5 that you did to Claude Fable and the best Google model. compare also with the best model out from Meta, Deepseek, and 1-2 others you think would be interesting. I have a key for OpenRouter you can use for these other model comparisons
>
> all of the above should be added as an epilogue to the original, I don't want to mess with what was run in class, but thought the additions would be useful for discussion. give me the tokens used for the epilogue, cost of each run (go back and add the cost of each run to the ones run in class if you did not put that already) how long each run took, etc. 
>
> if possible compare to GPT-4 and GPT-3.5 if those are available in the epilogue

## 7. 14:26

> do you need my openrouter key?

## 8. 15:06

> tell me how to add a working OpenRouter key for you

## 9. 15:08

> ok i added the key as you suggested check it

## 10. 15:21

> how does the LLM version work, are you sending several images with it or do you compile one large image with all the options and it just returns the correct answer? provide a visual explanation of what is being sent to the LLM in the readme documents along with the prompt so it explains exactly the process it goes through for the LLM solutions

## 11. 15:24

> would it be possible to do any of these in batch mode? if so how would that impact the cost? if possible, estimate the batch mode cost for these models. do not actually run batch mode if you haven't already, just estimate

## 12. 15:28

> do some error analysis on the ones that are commonly incorrect by the LLM models. are all the LLMs getting the same questions incorrect? why are they choosing the answers they are choosing? what was the correct answer and why were the models missing the correct answer?
>
> how long do you expect training to take? do I need to leave my machine at the office overnight

## 13. 15:35

> write up a narrative of all of these experiments in the readme for this repo. provide visual examples of the ravens problems, showing an easy, medium, and hard. what did the original code try to do? what did claude code attempt for the non-llm solutions and why did it choose those methods? what were the results from the LLM versions including the accuracy, cost, and time they took. what were common mistakes the models made? give visual examples with the ravens matrix included so people can see along with the explanations. keep in mind these are undergraduate business students with no training in ML, AI or anything, assume they are smart buy have no prior knowledge of ML or Raven's or reasoning etc. Be clear about the LLM version solved the problem, what exactly was shown to the LLM and what did the LLM return. Be concise but write at a standard undergraduate level. include all of this in the readme like it was an experiment (it was) and a teachable experiment. at the end you can give a section on technical notes for students who are technical and curious to try things out, point them at parts of the code and give them a software architecture and data architecture on how things worked. push it all out to github for them to see.

## 14. 15:36

> you don't need to respond in line here, build all the explanations into the readme narrative

## 15. 15:46

> I put a CSV file with student predictions on whether you would be able to get code running by the end of class 1:50PM and how many would be correct from the LLM out of 96. I accidentally put the same question twice you will see so you can ignore one question. do analysis of who got the prediction correct for the "will it finish" and then give the top five students who guessed the closest for the non-llm version. just use your best judgement on those wihtout asking me how to judge it. put the top 5 student names in the readme as the winners with a big gold, silver, bronze, etc.

## 16. 15:47

> finally, put a .md file with all of the prompts I have given you for this whole project so far in the repository and push it. call it PROMPT_HISTORY.md or something similar, link to it in the readme

## 17. 17:50

> add total cost of the experiment, be more specific about what kind of approach it used here- Modern program, no language model, [Image #2] this image should be broken down, I want to know exactly how many files it sent, like 1 whole image then all cells and options? it's not clear to me. movethe "SAME PAYLOAD DIFFERENT BILL" out of the description of what was sent so it is clear what was sent [Image #3]. for contestant 2 be more specific aboat the approach, see the PDF in the ravensdemo directory about AI techniques. was this expert? knowledge based? tie it to course concepts a bit more so it is clear. what was the original "student" approach? in the comparison table highlight the most accurate, highest cost, and longest time. make a small graph comparing all the OpenAI sorted by their release date. put their release date along on the graph to show. for the whole table do your best to add the release date of each model as a new column. is it possible to show how many tokens claude code spent on the in-class experiment as well as the epilogue? show that as a section. add a section with some explanation of how the LLM is doing what it is doing, at an undergrad reading level. give links to more details such as a vision transformer, embeddings, etc. include this prompt in the prompt history file.

## 18. 17:52

> rather than a commit make this a pull request, then push the PR. I want to show a pull request to students
