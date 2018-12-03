import sys

number_of_digits = int(input())

attempts = 10

start_at  = 1
finish_at = number_of_digits
first_launch = True
old_finish_at = finish_at

middle = (start_at + finish_at) // 2

while start_at != finish_at:

    old_middle = middle
    middle = (start_at + finish_at) // 2
    if first_launch:
        finish_at = finish_at // 2
        first_launch = False

    print('? ', range(start_at, finish_at))
    answer = input()

    if answer == 'Odd':
        old_finish_at = finish_at
        finish_at = middle
    elif answer == 'Even':
        start_at = old_middle + 1
        finish_at = old_finish_at



print(start_at)



# go_right  = False
#
# middle = int((finish_at + start_at) / 2)
#
# for attempt in range(attempts):
#     print('# ', attempt + 1)
#
#
#
#     if not go_right:
#         ask_range = range(start_at, middle)
#     else:
#         ask_range = range(middle+1, finish_at)
#
#     print('? ', ask_range)
#     answer = input()
#
#     middle = int((finish_at + start_at) / 2)
#
#     #left
#     if answer == 'Odd':
#         go_right = False
#         finish_at = middle
#     #right
#     elif answer == 'Even':
#         go_right = True
#         start_at  = middle + 1
#
#     print(start_at, middle, finish_at)
