using Plots
plotlyjs()
##
# base image from dataset load on detect.jl
baseimg = convert2RGB(img_[:,:,:,1] |> cpu) |> copy

# box shape
rectangle(x,y,w,h) = Shape(x .+ [0, w, w, 0], y .+ [0,0,h,h])


gtboxplot = rectangle(gtbox[:,:,1,1]...)
dtboxplot = rectangle(dtbox[1:4]...)

# tapline pixels size(w,h)
# from detect.jl
gtpxlplot = gtpxl[:,:,1,1];
dtpxlplot = dttapline[:,:,1];

gtpxl_px = map(findall(Bool.(gtpxlplot))) do x
    x = (Float32(x.I[2]), Float32(x.I[1]))
end

dtpxl_px = map(findall(Bool.(dtpxlplot))) do x
    x = (Float32(x.I[2]), Float32(x.I[1]))
end

# tapline endpoints
gtpxl_px[1] = (gtbox[:,:,1,1][1], gtbox[:,:,1,1][2])
gtpxl_px[end] = (gtbox[:,:,1,1][1] + gtbox[:,:,1,1][3], gtbox[:,:,1,1][2] + gtbox[:,:,1,1][4])

dtpxl_px[1] = (dtbox[:,:,1,1][1], dtbox[:,:,1,1][2])
dtpxl_px[end] = (dtbox[:,:,1,1][1] + dtbox[:,:,1,1][3], dtbox[:,:,1,1][2] + dtbox[:,:,1,1][4])


# filter dtpxl_px witin dtbox[1:4]
dtpxl_px = filter(dtpxl_px) do x
    a=x
    (a[1] >= dtpxl_px[1][1] && a[1] <= dtpxl_px[end][1]) && 
    (a[2] >= dtpxl_px[1][2] && a[2] <= dtpxl_px[end][2])
end


#
# start plot
plot(baseimg, size=(400,400))
# gt-box
plot!(gtboxplot,  
    linewidth=3,
    linestyle = :dot,
    linecolor=:red, 
    fillalpha=0.1, 
    legend=false
)
# dt-box
plot!(dtboxplot,  
    linewidth=2,
    linecolor=:yellow, 
    fillalpha=0.1, 
    legend=false
)

# tapline
# gt-pxl
plot!(gtpxl_px, 
    markershape=:circle,
    markersize=2,
    markeralpha=0.7,
    markercolor=:red,
    markerstrokewidth=2,
    markerstrokealpha=0.2,
    markerstrokecolor=:red,
    markerstrokestyle=:line,# :dot
    linewidth=2,
    linealpha = 0.5,
    linecolor=:red,
)

plot!(dtpxl_px, 
    markershape=:circle,
    markersize=2,
    markeralpha=0.7,
    markercolor=:yellow,
    markerstrokewidth=2,
    markerstrokealpha=0.2,
    markerstrokecolor=:yellow,
    markerstrokestyle=:line,# :dot
    linewidth=2,
    linecolor=:yellow, 
    linealpha = 0.5
)