import turtle
n = int(input("5이상의 자연수를 입력하세요: "))
t = turtle.Turtle()
for _ in range(n):
    t.forward(200)
    t.right(180-180/n)
turtle.exitonclick()