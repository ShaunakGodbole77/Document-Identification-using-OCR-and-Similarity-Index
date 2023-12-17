from thefuzz import fuzz, process

answer1 = "Today is a very good day"
answer2 = "It is a great day for me today"

print(fuzz.token_sort_ratio(answer1,answer2))
