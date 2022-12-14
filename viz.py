import plotly.graph_objects as go
import plotly.express as px
import pyefd
from plotly.subplots import make_subplots
import pyvista as pv
import numpy as np
from scipy.spatial import KDTree
import pandas as pd
from utils import approximate_one_param

pv.global_theme.font.color = "black"

def get_square_wave_fig(x,y,L,an,bn,n_terms=100):
    fig = go.Figure()
    
    fig.update_layout(
        autosize=False,
        width=1100,
        height=600)

    fig.add_trace(
        go.Scatter(
            visible=True,
            line=dict(color="black", width=2),
            name="Original function", 
            x=x,
            y=y))

    for terms in np.arange(1, n_terms+1, 1):
        fig.add_trace(
            go.Scatter(
                visible= terms == 1,
                line=dict(color="#00CED1", width=2),
                name=f"Fourier approximation - {terms} term(s)", 
                x=x,
                y=sum([an(k)*np.cos(2.*np.pi*k*x/L)+bn(k)*np.sin(2.*np.pi*k*x/L) for k in range(1,terms+1)])))

    steps = []
    for i in range(1,len(fig.data)):
        step = dict(
            method="update",
            label=i,
            args=[{"visible": [False] * len(fig.data)}],
        )
        step["args"][0]["visible"][i] = True
        step["args"][0]["visible"][0] = True
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Terms: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders
    )

    return fig

def get_polar_contour_fig(rad, theta):
    fig = go.Figure()
    fig.update_layout(
        autosize=False,
        width=400,
        height=400)
    fig.add_trace(go.Scatterpolar(
            r=rad,
            theta=theta,
            mode="lines",
            thetaunit = "radians",
        ))
    fig.update_layout(polar = dict(radialaxis = dict(showticklabels = False)))
    return fig

def get_square_cartesian_vs_polar_fig(x,y,r,theta):
    fig = make_subplots(
        rows=1, cols=2, 
        specs=[[{},{"type":"polar"}]],
        column_widths=[0.45,0.55])

    fig.update_layout(
        autosize=False,
        width=900,
        height=600)

    fig.add_trace(
        go.Scatter(
            visible=True,
            line=dict(color="black", width=2),
            x=x,
            y=y,
            showlegend=False),
        row=1, col=1)

    fig.update_xaxes(range=[x.min()-0.2, x.max()+0.2],row=1,col=1)
    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1,
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatterpolar(
            visible=True,
            line=dict(color="black", width=2),
            thetaunit = "radians",
            theta=theta,
            r=r,
            showlegend=False),
        row=1, col=2)

    fig.update_polars(radialaxis=dict(range=[0, 1.5]))
    fig.update_layout(polar = dict(radialaxis = dict(showticklabels = False)))

    return fig

def get_one_param_polar_fig(theta, rad, x, y):

    fig = make_subplots(rows=1, cols=2,
                        column_widths=[0.6,0.4])
    
    sorted_idxs = np.argsort(theta)
    theta_sorted = theta[sorted_idxs]
    rad_sorted = rad[sorted_idxs]

    fig.update_layout(
        autosize=False,
        width=1100,
        height=600)


    fig.add_trace(
        go.Scatter(
            visible=True,
            line=dict(color="black", width=2),
            name="Original function                ", 
            x=x,
            y=y),
        row=1, col=1)

    fig.update_xaxes(range=[x.min()-0.2, x.max()+0.2],row=1,col=1)
    fig.update_yaxes(range=[y.min()-0.2, y.max()+0.2],row=1,col=1)

    fig.add_trace(
        go.Scatter(
            visible=True,
            mode="lines",
            line=dict(dash="dash", color="black", width=2),
            x=theta_sorted,
            y=rad_sorted,
            showlegend=False),
        row=1, col=2)

    thresholds = [-1, 5.0, 0.10, 0.05, 0.01, 0.005]
    thresholds_ncoeffs = [0,1,3,5,7,9]

    for t in thresholds[1:]:
        rfilt = approximate_one_param(rad, t)
        real_rfilt = np.real(rfilt)
        zx = real_rfilt*np.cos(theta)
        zy = real_rfilt*np.sin(theta)

        real_rfilt_sorted = real_rfilt[sorted_idxs]

        fig.add_trace(
            go.Scatter(
                visible=False,
                mode="lines+markers",
                line=dict(color="#00CED1", width=3),
                name=f"Fourier approximation", 
                x=zx,
                y=zy),
            row=1, col=1)
        
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="rgba(204, 102, 119, 0.7)", width=2),
                mode="lines",
                x=theta_sorted,
                y=real_rfilt_sorted,
                showlegend=True,
                name=r"$r(\theta)$"),
            row=1, col=2)

    steps = []
    n_traces = 2
    for i in range(0, len(fig.data)-1, n_traces):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}],
        )
        step["args"][0]["visible"][0] = True
        step["args"][0]["visible"][1] = True
        if i > 0:
            step["args"][0]["visible"][i:i+n_traces] = [True] * n_traces
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Terms: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders
    )

    fig["layout"]["xaxis2"]["title"]=r"$\theta$"

    fig.update_xaxes(tickvals=[-np.pi, 0, np.pi], 
                     ticktext=[r"$-\pi$", "0", r"$\pi$"],
                     range=[-np.pi-0.1, np.pi+0.1],
                     row=1, col=2)

    for i in range(len(thresholds_ncoeffs)):
        fig["layout"]["sliders"][0]["steps"][i]["label"] = f"{thresholds_ncoeffs[i]}"

    return fig

def get_two_param_2d_fig(coeffs, a0, c0, xy, n_points, n_terms, show_recon_err=False, set_aspect_ratio=False):
    fig = make_subplots(rows=1, cols=2,
                        column_widths=[0.6, 0.4])

    fig.update_layout(
        autosize=False,
        width=1100,
        height=600)

    fig.add_trace(
        go.Scatter(
            visible=True,
            line=dict(color="black", width=2),
            x=xy[:, 0],
            y=xy[:, 1],
            name="Original function           ",
            showlegend=True),
        row=1, col=1)

    if set_aspect_ratio:
        fig.update_xaxes(range=[xy[:,0].min() - 0.35, xy[:,0].max() + 0.2],row=1,col=1)
        fig.update_yaxes(range=[xy[:,1].min() - 0.35, xy[:,1].max() + 0.2],row=1,col=1)

    fig.update_layout(title="")

    recon_errs = []
    recon_errs.append(-1)

    dxy = np.diff(xy, axis=0)
    dxy = np.insert(dxy, 0, [0,0]).reshape(xy.shape[0],2)
    dt = np.sqrt((dxy ** 2).sum(axis=1))
    arcl = np.cumsum(dt)
    T = arcl[-1]
    t_orig = (2 * np.pi * arcl) / T

    fig.add_trace(
            go.Scatter(
                visible=True,
                line=dict(dash="dash", color="rgb(204, 102, 119)", width=2),
                mode="lines",
                showlegend=True,
                name="x(t)",
                x=t_orig,
                y=xy[:,0]),
            row=1, col=2)

    fig.add_trace(
        go.Scatter(
            visible=True,
            line=dict(dash="dash", color="rgb(136, 204, 238)", width=2),
            mode="lines",
            showlegend=True,
            name="y(t)",
            x=t_orig,
            y=xy[:,1]),
        row=1, col=2)

    t = np.linspace(0, 1.0, n_points)
    t_approx = t*2*np.pi

    xt = np.ones((n_points,)) * a0
    yt = np.ones((n_points,)) * c0

    for terms in range(n_terms):        
        xt += (coeffs[terms, 0] * np.cos(2 * (terms + 1) * np.pi * t)) + (
            coeffs[terms, 1] * np.sin(2 * (terms + 1) * np.pi * t)
        )
        yt += (coeffs[terms, 2] * np.cos(2 * (terms + 1) * np.pi * t)) + (
            coeffs[terms, 3] * np.sin(2 * (terms + 1) * np.pi * t)
        )

        if show_recon_err:
            points = np.asarray([xt,yt]).T
            points = KDTree(points)
            errors,ii = points.query(xy[0:-1,:])
            recon_errs.append(np.mean(errors))

        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="rgba(136, 204, 238, 0.7)", width=2),
                mode="lines",
                name="y(t) - approximation",
                showlegend=False,
                x=t_approx,
                y=yt),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="rgba(204, 102, 119, 0.7)", width=2),
                mode="lines",
                name="x(t) - approximation",
                showlegend=False,
                x=t_approx,
                y=xt),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="#00CED1", width=3),
                mode="lines+markers",
                name=f"Fourier approximation", 
                x=xt,
                y=yt),
        row=1, col=1)

    steps = []
    n_traces = 3
    for i in range(0, len(fig.data)-3, n_traces):
        if i > 0 and show_recon_err:
            title_label = f"Reconstruction error (MSE): {round(recon_errs[int(i/n_traces)],3)}"
        else:
            title_label = ""
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                {"title": title_label}],
        )
        step["args"][0]["visible"][0] = True
        step["args"][0]["visible"][1] = True
        step["args"][0]["visible"][2] = True
        if i > 0:
            step["args"][0]["visible"][i:i+n_traces] = [True] * n_traces
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Terms: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders
    )

    fig["layout"]["xaxis2"]["title"]=r"$t$"
    fig.update_xaxes(tickvals=[0, np.pi, 2*np.pi], 
                    ticktext=["0", r"$\pi$", r"$2\pi$"],
                    row=1, col=2)


    for i in range(n_terms):
        fig["layout"]["sliders"][0]["steps"][i]["label"] = f"{i}"

    return fig

def get_two_param_coeff_table(xy):
    save_terms = [1,8,16,24]
    saved_coeffs = []
    a0, c0 = pyefd.calculate_dc_coefficients(xy)
    for terms in save_terms:  
        coeffs = pyefd.elliptic_fourier_descriptors(xy, order=terms)
        saved_coeffs.append(coeffs)

    cols = []
    for i in range(24):
        cols.append(f"a{i}")
        cols.append(f"b{i}")
        cols.append(f"c{i}")
        cols.append(f"d{i}")

    cell_text = []
    for r in saved_coeffs:
        flat_r = r.flatten()
        n_blanks = 24*4 - flat_r.shape[0]
        vals = [f"{round(v,3)}" for v in flat_r]
        if n_blanks != 0:
            blanks = [""] * n_blanks
            vals.extend(blanks)
        cell_text.append(vals)

    df = pd.DataFrame(cell_text, columns=cols)
    df["n_terms"] = ["1 term", "8 terms", "16 terms", "24 terms"]
    df = df.set_index("n_terms")
    return df

def get_rotate_2d_fig(xy, xy_rotated):
    fig = make_subplots(rows=1, cols=2)

    fig.update_layout(
        autosize=False,
        width=900,
        height=400)

    fig.add_trace(
        go.Scatter(
            visible=True,
            line=dict(color="black", width=2),
            x=xy[:,0],
            y=xy[:,1],
            showlegend=False),
        row=1,col=1)

    fig.add_trace(
        go.Scatter(
            visible=True,
            line=dict(color="green", width=2),
            x=xy_rotated[:,0],
            y=xy_rotated[:,1],
            showlegend=False),
    row=1, col=2)
    return fig

def get_recon_mesh_plotter(orig_mesh, recon_mesh):
    plotter = pv.Plotter(window_size=[1000,300], shape=(1,2)) 
    plotter.subplot(0,0)
    plotter.add_mesh(orig_mesh, color="lightgray")
    plotter.subplot(0,1)
    plotter.add_mesh(recon_mesh, color="lightgray")
    plotter.set_background("white")
    return plotter

def interactive_latent_walk_plot(recon_meshes, latent_walk_meshes):
    p = pv.Plotter()

    p.add_mesh(recon_meshes[0], 
               render=False, 
               show_scalar_bar=False)

    p.show(auto_close=False, 
           interactive=True, 
           interactive_update=True)

    def update(i):
        i = int(i)
        p.add_mesh(recon_meshes[i], 
                   render=False,
                   show_scalar_bar=False)
        p.update()

    slider_labels = [str(i) for i in range(len(latent_walk_meshes))]
    lmax_slider = p.add_text_slider_widget(
        update, 
        slider_labels,
        0,
        pointa=(0.25, 0.9), 
        pointb=(0.75, 0.9), 
        event_type="always")

def get_pca_clust_latent_walk_fig(axes, walk_line_x, walk_line_y, labels):
    fig_clust = px.scatter(x=axes[:,0],
                           y=axes[:,1],
                           color=labels)

    fig_latent_walk = px.line(x=walk_line_x, 
                              y=walk_line_y,
                              markers=True)
    fig_latent_walk.update_traces(line_color="black")

    fig = go.Figure(data=fig_clust.data + fig_latent_walk.data)
    fig.add_annotation(
        x=walk_line_x[2],
        y=walk_line_y[2],
        ax=walk_line_x[1] - 0.2,
        ay=walk_line_y[1],
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        text="",
        showarrow=True,
        arrowhead=3,
        arrowsize=5,
        arrowwidth=1,
        arrowcolor="black")

    fig.update_layout(
        autosize=False,
        width=1100,
        height=600,
        xaxis_title="PC1", 
        yaxis_title="PC2")

    return fig

def get_pca_result_fig(axes, labels):
    fig = px.scatter(x=axes[:,0],
                    y=axes[:,1],
                    color=labels)
    fig.update_layout(
        autosize=False,
        width=1100,
        height=600,
        legend_title="shape",
        xaxis_title="PC1", 
        yaxis_title="PC2"
    )
    return fig

def interactive_reconstruction_plot(recon_errors, recon_meshes):
    pv.start_xvfb()
    lmaxes = np.arange(1,len(recon_errors)+1,1)
    p = pv.Plotter()
    p.set_background("white")
    
    recon_chart = pv.Chart2D(
        size=(0.46, 0.25), 
        loc=(0.02, 0.06), 
        x_label="Lmax", 
        y_label="Reconstruction error",
    )
    recon_line = recon_chart.line(lmaxes, recon_errors)
    recon_chart.background_color = (1.0, 1.0, 1.0, 0.4)
    p.add_chart(recon_chart)

    p.add_mesh(recon_meshes[0], 
               name="nuc", 
               render=False, 
               show_scalar_bar=False)

    p.show(auto_close=False, 
           interactive=True, 
           interactive_update=True)

    def update_lmax(l):
        l = int(l)
        recon_chart.clear("scatter")
        recon_chart.scatter([l],[0], style="s")
        p.add_mesh(recon_meshes[l], 
                   name="nuc", 
                   render=False,
                   show_scalar_bar=False)
        p.update()

    slider_labels = [str(i) for i in lmaxes.tolist()]
    lmax_slider = p.add_text_slider_widget(
        update_lmax, 
        slider_labels,
        0,
        pointa=(0.25, 0.9), 
        pointb=(0.75, 0.9), 
        event_type="always")