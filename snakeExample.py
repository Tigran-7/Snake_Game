import turtle
import time
import random

speed = 0.1
count = 0
high_count = 0
step = 20

# Set up the screen
screen = turtle.Screen()
screen.title("Game snake")
screen.bgcolor('grey')
screen.setup(800, 800)
screen.tracer(0)

# interface
turtle.speed(1)
turtle.pencolor('grey')
turtle.pensize(10)
turtle.goto(-350, -350)
turtle.pencolor('black')
turtle.goto(-350, 350)
turtle.goto(350, 350)
turtle.goto(350, -350)
turtle.goto(-345, -350)

# Snake  head
snake = turtle.Turtle()
snake.speed(0)
snake.shape('circle')
snake.color('red') # +X argument
snake.penup()
snake.goto(0,0)
snake.direction = 'stop'

# Snake food
ball = turtle.Turtle()
ball.speed(0)
ball.shape("circle")
ball.color('yellow')
ball.penup()
bx = random.randint(-14, 14) * 20
by = random.randint(-14, 14) * 20
print(bx, by)
ball.goto(bx, by)
ball.resizemode('sircle')
segments = []

# Pen
pen = turtle.Turtle()
pen.speed(0)
pen.shape('square')
pen.color('white')
pen.penup()
pen.hideturtle()
pen.goto(-360, 360)
pen.write('Score: 0  High Score: 0', align="left", font=("", 24, ''))

# Functions


def go_up():
    if snake.direction != "down":
        snake.direction = 'up'


def go_down():
    if snake.direction != "up":
        snake.direction = 'down'


def go_left():
    if snake.direction != "right":
        snake.direction = 'left'


def go_right():
    if snake.direction != "left":
        snake.direction = 'right'


def move():
    if snake.direction == 'up':
        y = snake.ycor()
        snake.sety(y + step)

    if snake.direction == 'down':
        y = snake.ycor()
        snake.sety(y - step)

    if snake.direction == 'left':
        x = snake.xcor()
        snake.setx(x - step)

    if snake.direction == 'right':
        x = snake.xcor()
        snake.setx(x + step)

# def move2(bx, by):
#    if snake.xcor() > bx:
#       snake.setx(snake.xcor() - step)
#    if snake.xcor() < bx:
#        snake.setx(snake.xcor() + step)
#    if snake.xcor() == bx:
#        if snake.ycor() > by:
#            snake.sety(snake.ycor() - step)
#        if snake.ycor() < by:
#            snake.sety(snake.ycor() + step)

# def move3(bx, by):
#    snake.speed(1)
#    snake.dot(12, 'black')
#    snake.onclick(time.sleep(1), 'Space', 10)
#    move2(bx, by)

# Keyboard bindings
screen.listen()
screen.onkeypress(go_up, 'Up')
screen.onkeypress(go_down, 'Down')
screen.onkeypress(go_right, 'Right')
screen.onkeypress(go_left, 'Left')
# Main game loop
while True:
    screen.update()

    # Check for a collision with the border
    if snake.xcor()> 330 or snake.xcor()<-330 or snake.ycor()>330 or snake.ycor()<-330:
        snake.goto(0, 0)
        snake.direction = "stop"
        print('ball coordinates ', bx, by)

        # Hide the segments
        for segment in segments:
            segment.goto(450, 450)

        # Clear the segments list
        segments = []

        # Reset the delay
        speed -= 0.00001

        # Score
        count = 0

        pen.clear()
        pen.write("Score: {} High Score: {}".format(count, high_count), align='left', font=("", 24, ""))

    # Check for a collision with the food
    if snake.distance(ball) < 20:
        bx = random.randint(-15, 15) * 20
        by = random.randint(-15, 15) * 20

        ball.goto(bx, by)

        # Add a segment
        new_segment = turtle.Turtle()
        new_segment.speed(0)
        new_segment.shape("circle")
        new_segment.color("red")
        new_segment.penup()
        segments.append(new_segment)

        # Shorten the delay
        if speed > 0.0001:
            speed -= 0.0001

        # Increase the score
        count += 10

        if count > high_count:
            high_count = count

        pen.clear()
        pen.write("Score: {}  High Score: {}".format(count, high_count), align='left', font=('', 24, ''))

    # Move the end segments first in reverse order
    for index in range(len(segments)-1, 0, -1):
        x = segments[index-1].xcor()
        y = segments[index-1].ycor()
        segments[index].goto(x, y)

    # Move segment 0 to where the head is
    if len(segments) > 0:
        x = snake.xcor()
        y = snake.ycor()
        segments[0].goto(x, y)

    # move2(bx, by)
    move()
    # Check for head collision with the body segments
    for segment in segments:
        if segment.distance(snake) < 20:
            time.sleep(1)
            snake.goto(0, 0)
            snake.direction = 'stop'

            # Hide the segments
            for seg in segments:
                seg.goto(450, 450)

            # Clear the segments list
            segments = []

            # Reset the delay
            speed = 0.1

            # Update the score display
            pen.clear()
            pen.write("Score: {} High_Score".format(count, high_count), align="left", font=("", 24, ""))
    time.sleep(speed)






