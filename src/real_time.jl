using GLMakie

function mandelbrot(x, y)
    RGBf(rand(),rand(),rand())
end


N = 50
xmin = 1
xmax = 1024
ymin = 1
ymax = 1024

x = collect(xmin:xmax)#LinRange(-2, 1, 200)
y = collect(ymin:ymax)#LinRange(-1.1, 1.1, 200)
matrix = mandelbrot.(x, y')
fig, ax, hm = image(x, y, matrix)


# we use `record` to show the resulting video in the docs.
# If one doesn't need to record a video, a normal loop works as well.
# Just don't forget to call `display(fig)` before the loop
# and without record, one needs to insert a yield to yield to the render task
display(fig)
for i in 1:N
    hm[3] = mandelbrot.(x, y') # update data

     yield()
end
