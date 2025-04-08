import plotly.graph_objects as go
import numpy as np


# Generate sample data
np.random.seed(42)
x = np.random.randn(100)
y = np.random.randn(100)
z = np.random.randn(100)

# Create the 3D scatter plot with vibrant markers
fig = go.Figure(
    data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=8,
            color=z,             # Colors based on z values
            colorscale='Rainbow',# Vibrant colorscale
            opacity=1.0,
            colorbar=dict(title='Z')  # Optional: color bar for reference
        )
    )],
    layout=go.Layout(
        title="Auto-Rotating 3D Scatter Plot on a Black Background",
        paper_bgcolor='black',  # Full page background color
        plot_bgcolor='black',   # Plot area background color
        scene=dict(
            bgcolor='black',    # Background of the 3D scene
            xaxis=dict(
                showbackground=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title=''
            ),
            yaxis=dict(
                showbackground=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title=''
            ),
            zaxis=dict(
                showbackground=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title=''
            ),
            camera=dict(
                # Set an initial view for the camera.
                eye=dict(x=2, y=2, z=1)
            )
        ),
        # Add a play button to trigger the rotation animation.
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "buttons": [{
                "label": "Play",
                "method": "animate",
                "args": [None, {
                    "frame": {"duration": 100, "redraw": True},
                    "fromcurrent": True,
                    "transition": {"duration": 0}
                }]
            }]
        }]
    )
)

# Create frames that rotate the camera around the z-axis.
frames = []
n_frames = 360  # Number of frames for a full 360° rotation.
for theta in np.linspace(0, 2*np.pi, n_frames):
    # Rotate the camera in the xy-plane around the z-axis.
    eye = dict(x=2 * np.cos(theta), y=2 * np.sin(theta), z=1)
    frames.append(go.Frame(layout=dict(scene=dict(camera=dict(eye=eye)))))

fig.frames = frames

# Display the plot; click "Play" to start the auto-rotation.
fig.show()
