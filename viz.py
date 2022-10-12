import plotly.graph_objects as go
import plotly.express as px
import pyefd
from plotly.subplots import make_subplots
import pyvista as pv
import numpy as np
from scipy.spatial import KDTree
from scipy import interpolate
import glob


def get_square_wave_fig(x,y,L,an,bn,n_terms=100):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            visible=True,
            line=dict(color="#FFA500", width=2),
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


def get_recon_mesh_plotter(orig_mesh, recon_mesh):
    plotter = pv.Plotter(window_size=[900,400], shape=(1,2)) 
    plotter.subplot(0,0)
    plotter.add_mesh(orig_mesh, color='lightgray')
    plotter.subplot(0,1)
    plotter.add_mesh(recon_mesh, color='lightgray')
    plotter.set_background('white')
    return plotter

def interactive_latent_walk_plot(recon_meshes, latent_walk_meshes):
    p = pv.Plotter()
#     p.set_background('white')

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
        event_type='always')

def get_polar_contour_fig(rad, theta):
    fig = go.Figure()
    fig.update_layout(
        autosize=False,
        width=400,
        height=400)
    fig.add_trace(go.Scatterpolar(
            r=rad,
            theta=theta,
            mode='lines',
            thetaunit = "radians",
        ))
    fig.update_layout(polar = dict(radialaxis = dict(showticklabels = False)))
    return fig

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
        xref='x',
        yref='y',
        axref='x',
        ayref='y',
        text='',
        showarrow=True,
        arrowhead=3,
        arrowsize=5,
        arrowwidth=1,
        arrowcolor='black',
        )
    return fig

def get_one_param_polar_fig(theta, rad, x, y):
    def approximate_one_param(z, thresh):
        n = len(z)
        zhat = np.fft.fft(z, n)
        PSD = np.real(zhat * np.conj(zhat)/n)
        indices = PSD > thresh
        PSDclean = PSD * indices
        zhat = indices * zhat
        zfilt = np.fft.ifft(zhat)
        return zfilt

    fig = make_subplots(rows=1, cols=2,
                        column_widths = [0.6,0.4])

    fig.update_layout(
        autosize=False,
        width=500,
        height=300)


    fig.add_trace(
        go.Scatter(
            visible=True,
            line=dict(color="#FFA500", width=2),
            name="Original function          ", 
            x=x,
            y=y),
        row=1, col=1)

    fig.update_xaxes(range=[x.min()-0.2, x.max()+0.2],row=1,col=1)
    fig.update_yaxes(range=[y.min()-0.2, y.max()+0.2],row=1,col=1)

    fig.add_trace(
        go.Scatter(
            visible=True,
            mode="lines",
            line=dict(dash="dash", color="#FFA500", width=2),
            x=np.arange(0,len(rad),1),
            y=rad,
            showlegend=False),
        row=1, col=2)

    thresholds = [5.0, 0.10, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
    thresholds_ncoeffs = [1,3,5,7,9,13,17,29,37]
    for t in thresholds:
        rfilt = approximate_one_param(rad, t)
        real_rfilt = np.real(rfilt)
        zx = real_rfilt*np.cos(theta)
        zy = real_rfilt*np.sin(theta)

        fig.add_trace(
            go.Scatter(
                visible=False,
                mode="lines",
                line=dict(color="#00CED1", width=2),
                name=f"Fourier approximation", 
                x=zx,
                y=zy),
            row=1, col=1)
        
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="#00CED1", width=2),
                mode="lines+markers",
                x=np.arange(0,len(real_rfilt),1),
                y=real_rfilt,
                showlegend=False),
            row=1, col=2)

    steps = []
    n_traces = 2
    for i in range(0, len(fig.data)-2, n_traces):
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

    for i in range(len(thresholds)):
        fig["layout"]["sliders"][0]["steps"][i]["label"] = f"{thresholds_ncoeffs[i]}"

    return fig

def get_one_param_polar_fig_prev(theta, rad, theta_interp, rad_interp, an, bn):

    fig = make_subplots(rows=1, cols=2,
                        column_widths = [0.7,0.3],
                        specs=[[{"type":"polar"},{}]])

    fig.add_trace(
        go.Scatterpolar(
            visible=True,
            line=dict(color="#FFA500", width=2),
            name="Original function", 
            thetaunit = "radians",
            theta=theta,
            r=rad),
        row=1, col=1)
    fig.update_layout(polar = dict(radialaxis = dict(showticklabels = False)))


    fig.add_trace(
        go.Scatter(
            visible=True,
            mode="lines+markers",
            line=dict(color="#FFA500", width=2),
            x=theta[1:],
            y=rad[1:],
            showlegend=False),
        row=1, col=2)
    fig.update_layout(
        xaxis_title=r"$\theta$",
        yaxis_title=r"$r(\theta)$"
    )

    for terms in np.arange(1, 101, 1):
        
        r_approx = an(0)/2 + sum([an(k)*np.cos(k*theta_interp)+bn(k)*np.sin(k*theta_interp) for k in range(1,terms+1)])
        
        fig.add_trace(
            go.Scatterpolar(
                visible=(terms==1),
                mode="lines",
                line=dict(color="#00CED1", width=2),
                name=f"Fourier approximation - {terms} term(s)", 
                theta=theta_interp,
                thetaunit="radians",
                r=r_approx),
            row=1, col=1)
        
        fig.add_trace(
            go.Scatter(
                visible=(terms==1),
                line=dict(color="#00CED1", width=2),
                mode="lines+markers",
                x=theta_interp,
                y=r_approx,
                showlegend=False),
            row=1, col=2)

    steps = []
    for i in range(2,int(len(fig.data)) - 2):
        step = dict(
            method="update",
            label=i-1,
            args=[{"visible": [False] * len(fig.data)}],
        )
        step["args"][0]["visible"][i] = True
        step["args"][0]["visible"][i+1] = True
        step["args"][0]["visible"][0] = True
        step["args"][0]["visible"][1] = True
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


def reconstruct_contour(coeffs, locus=(0, 0), num_points=300):
    """Returns the contour specified by the coefficients.

    :param coeffs: A ``[n x 4]`` Fourier coefficient array.
    :type coeffs: numpy.ndarray
    :param locus: The :math:`A_0` and :math:`C_0` elliptic locus in [#a]_ and [#b]_.
    :type locus: list, tuple or numpy.ndarray
    :param num_points: The number of sample points used for reconstructing the contour from the EFD.
    :type num_points: int
    :return: A list of x,y coordinates for the reconstructed contour.
    :rtype: numpy.ndarray

    """
    t = np.linspace(0, 1.0, num_points)
    # Append extra dimension to enable element-wise broadcasted multiplication
    coeffs = coeffs.reshape(coeffs.shape[0], coeffs.shape[1], 1)

    orders = coeffs.shape[0]
    orders = np.arange(1, orders + 1).reshape(-1, 1)
    order_phases = 2 * orders * np.pi * t.reshape(1, -1)

    xt_all = coeffs[:, 0] * np.cos(order_phases) + coeffs[:, 1] * np.sin(order_phases)
    yt_all = coeffs[:, 2] * np.cos(order_phases) + coeffs[:, 3] * np.sin(order_phases)

    xt_all = xt_all.sum(axis=0)
    yt_all = yt_all.sum(axis=0)
    xt_all = xt_all + np.ones((num_points,)) * locus[0]
    yt_all = yt_all + np.ones((num_points,)) * locus[1]

    reconstruction = np.stack([xt_all, yt_all], axis=1)
    return reconstruction, order_phases[0]


def get_two_param_2d_fig(coeffs, a0, c0, xy, n_points, n_terms, show_recon_err=False, set_aspect_ratio=False):
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.6, 0.4])

    fig.update_layout(
    autosize=False)

    fig.add_trace(
        go.Scatter(
            visible=True,
            line=dict(color="#FFA500", width=2),
            x=xy[:, 0],
            y=xy[:, 1],
            name="Original function      ",
            showlegend=True),
        row=1, col=1)

    if set_aspect_ratio:
        fig.update_xaxes(range=[xy[:,0].min(), xy[:,0].max()],row=1,col=1)

    fig.update_layout(title=" ")

    recon_errs = []
    recon_errs.append(-1)

    t = np.linspace(0, 1.0, n_points)
    t_approx = t*2*np.pi
    t_orig = np.linspace(0, 1.0, xy.shape[0])
    t_orig = t_orig*2*np.pi
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
                name="y(t)",
                x=t_approx,
                y=yt),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(dash="dash", color="rgb(136, 204, 238)", width=2),
                mode="lines",
                showlegend=False,
                x=t_orig,
                y=xy[:,1]),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="rgba(204, 102, 119, 0.7)", width=2),
                mode="lines",
                name="x(t)",
                x=t_approx,
                y=xt),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(dash="dash", color="rgb(204, 102, 119)", width=2),
                mode="lines",
                showlegend=False,
                x=t_orig,
                y=xy[:,0]),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="#00CED1", width=1),
                mode="lines+markers",
                name=f"Fourier approximation", 
                x=xt,
                y=yt),
        row=1, col=1)

    steps = []

    n_traces = 5
    for i in range(0, len(fig.data)-1, n_traces):
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

    # fig.update_xaxes(
    #     tickmode = 'array',
    #     tickvals = [0, np.pi, 2*np.pi],
    #     ticktext = [0, "$\pi$", "$2\pi$"],
    #     row=1, col=2
    # )

    for i in range(n_terms):
        fig["layout"]["sliders"][0]["steps"][i]["label"] = f"{i}"

    return fig


def get_two_param_2d_fig_with_eqs(coeffs, a0, c0, xy, n_points):
    N = coeffs.shape[0]
    N_half = int(np.ceil(N / 2))
    t = np.linspace(0, 1.0, n_points)
    xt = np.ones((n_points,)) * a0
    yt = np.ones((n_points,)) * c0

    approx_xs = []
    approx_ys = []

    fig = go.Figure()

    eqs_x = []
    eqs_y = []
    eq_xtheta = "x(\\theta) = \\frac{{a0}}{2} ".replace("a0", f"{round(a0,1)}")
    eq_ytheta = "y(\\theta) = \\frac{{c0}}{2} ".replace("c0", f"{round(c0,1)}")

    fig.add_trace(
        go.Scatter(
            visible=True,
            line=dict(color="#FFA500", width=2),
            name="Original function", 
            x=xy[:, 0],
            y=xy[:, 1]))

    fig.update_layout(title=r"${{eq_x}} \\ {{eq_y}}$".replace("eq_x", eq_xtheta).replace("eq_y", eq_ytheta))


    for terms in range(N):
        xt += (coeffs[terms, 0] * np.cos(2 * (terms + 1) * np.pi * t)) + (
            coeffs[terms, 1] * np.sin(2 * (terms + 1) * np.pi * t)
        )
        yt += (coeffs[terms, 2] * np.cos(2 * (terms + 1) * np.pi * t)) + (
            coeffs[terms, 3] * np.sin(2 * (terms + 1) * np.pi * t)
        )

        eq_xtheta = "{{prev_eq}} + {{c1}} \\cos({{terms}} \\pi * \\theta) + {{c2}} \\sin({{terms}} \\pi * \\theta)"\
                        .replace("c1", f"{round(coeffs[terms, 0],1)}")\
                        .replace("c2", f"{round(coeffs[terms, 1],1)}")\
                        .replace("terms", f"{terms+1}")\
                        .replace("prev_eq", eq_xtheta)
        eqs_x.append(eq_xtheta)

        eq_ytheta = "{{prev_eq}} + {{c3}} \\cos({{terms}} \\pi * \\theta) + {{c4}} \\sin({{terms}} \\pi * \\theta)"\
                    .replace("c3", f"{round(coeffs[terms, 2],1)}")\
                    .replace("c4", f"{round(coeffs[terms, 3],1)}")\
                    .replace("terms", f"{terms+1}")\
                    .replace("prev_eq", eq_ytheta)
        eqs_y.append(eq_ytheta)

        approx_ys.append(yt)
        fig.add_trace(
            go.Scatter(
                visible=(terms == 0),
                line=dict(color="#00CED1", width=2),
                name=f"Fourier approximation - {terms} term(s)", 
                x=xt,
                y=yt))

    steps = []
    for i in range(0,len(fig.data) - 1):
        step = dict(
            method="update",
            label=i,
            args=[{"visible": [False] * len(fig.data)},
                  {"title": r"${{eqx}} \\ {{eqy}}$".replace("eqx", eqs_x[i]).replace("eqy", eqs_y[i])}],
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
        sliders=sliders,
        title={'font': {'size': 7},
              'x':0}
    )
    return fig

def get_two_param_2d_fig_(coeffs, a0, c0, xy, n_points):
    N = coeffs.shape[0]
    N_half = int(np.ceil(N / 2))
    t = np.linspace(0, 1.0, n_points)
    xt = np.ones((n_points,)) * a0
    yt = np.ones((n_points,)) * c0

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            visible=True,
            line=dict(color="#FFA500", width=2),
            name="Original function", 
            x=xy[:, 0],
            y=xy[:, 1]))

    for terms in range(N):
        xt += (coeffs[terms, 0] * np.cos(2 * (terms + 1) * np.pi * t)) + (
            coeffs[terms, 1] * np.sin(2 * (terms + 1) * np.pi * t)
        )
        yt += (coeffs[terms, 2] * np.cos(2 * (terms + 1) * np.pi * t)) + (
            coeffs[terms, 3] * np.sin(2 * (terms + 1) * np.pi * t)
        )

        fig.add_trace(
            go.Scatter(
                visible=(terms == 0),
                line=dict(color="#00CED1", width=2),
                name=f"Fourier approximation - {terms} term(s)", 
                x=xt,
                y=yt))

    steps = []
    for i in range(0,len(fig.data) - 1):
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
        sliders=sliders,
        title={'font': {'size': 7},
              'x':0}
    )
    return fig

def write_3d_reconstruction_gif(gt_mesh, recon_errors, out_file="output/lmax_reconstruction_nucleus.gif"):
    mesh_files = sorted(glob.glob("output/recon-0*.vtk"))

    plotter = pv.Plotter(shape=(1,2), notebook=False, off_screen=True)
    plotter.open_gif(out_file)

    plotter.subplot(0,0)
    plotter.add_mesh(gt_mesh, color="lightgray", show_scalar_bar=False)
    recon_mesh = pv.read(mesh_files[0])
    plotter.subplot(0,1)
    plotter.add_mesh(recon_mesh)
    label = plotter.add_text(f"Reconstruction error: {recon_errors[0]}", 
                            position='upper_right', 
                            font_size=20)
    plotter.set_background('white')
    plotter.write_frame()

    for i,f in enumerate(mesh_files[1:]):
        plotter.clear()
        plotter.subplot(0,0)
        plotter.add_mesh(gt_mesh, color="lightgray", show_scalar_bar=False)

        plotter.subplot(0,1)
        mesh_i = pv.read(f)
        plotter.add_mesh(mesh_i)
        plotter.add_text(f"Reconstruction error: {recon_errors[i]}", 
                        position='upper_right', 
                        font_size=20)
        plotter.write_frame()
        plotter.write_frame()
        plotter.write_frame()

    plotter.close()
    print(f"Wrote {out_file}")

import numpy as np
import pyvista as pv
pv.global_theme.font.color = 'black'
pv.start_xvfb()

def interactive_reconstruction_plot(recon_errors, recon_meshes):
    lmaxes = np.arange(1,len(recon_errors)+1,1)
    p = pv.Plotter()
    p.set_background('white')
    
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
        event_type='always')
        