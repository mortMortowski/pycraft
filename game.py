# IMPORTS
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import glm
import time
from PIL import Image
import cProfile
import random
import os
import sys

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

CHUNK_SIZE = 16
block_scale_factor = 1.25

# CLASSES
class Block:
    def __init__(self, block_id, block_type, color, position, texture):
        self.block_id = block_id # Id of the block
        self.block_type = block_type # Type of block, e.g. grass, cobblestone
        self.color = color # Color of the block. Later will be replaced by texture
        self.position = position # Position in 3D space of the block
        self.texture = texture # Block texture

class Player:
    def __init__(self, position, width=0.5, height=4):
        self.position = glm.vec3(position)
        self.width = width
        self.height = height
        self.velocity = glm.vec3(0.0, 0.0, 0.0)
        self.is_grounded = False
        self.gravity = 15
        self.jump_strength = 7.0

    def apply_gravity(self, deltaTime):
        if not self.is_grounded:
            if(deltaTime < 1): # Initial value is very big so discard it
                self.velocity[1] -= self.gravity * deltaTime     

    def jump(self):
        if self.is_grounded:
            self.velocity[1] = self.jump_strength
            self.is_grounded = False

    def update_position(self, deltaTime):
        self.position += self.velocity * deltaTime

class Frustum:
    def __init__(self, proj_matrix, view_matrix):
        # Combine projection and view matrices
        clip_matrix = np.dot(proj_matrix, view_matrix)
        self.planes = self._extract_planes(clip_matrix)

    def _extract_planes(self, clip_matrix):
        planes = np.zeros((6, 4)) # Six planes, each with four coefficients (Ax + By + Cz + D = 0)

        # Right plane
        planes[0] = clip_matrix[3] - clip_matrix[0]

        # Left plane
        planes[1] = clip_matrix[3] + clip_matrix[0]

        # Bottom plane
        planes[2] = clip_matrix[3] + clip_matrix[1]

        # Top plane
        planes[3] = clip_matrix[3] - clip_matrix[1]

        # Far plane
        planes[4] = clip_matrix[3] - clip_matrix[2]

        # Near plane
        planes[5] = clip_matrix[3] + clip_matrix[2]

        # Normalize the planes (for distance calculations)
        norms = np.linalg.norm(planes[:, :3], axis=1, keepdims=True) # Norm of the (A, B, C) part of the plane equation
        planes /= norms

        return planes
    
    def is_box_in_frustum(self, box_min, box_max):
        # Define the 8 corners of the AABB (Axis-Aligned Bounding Box)
        corner_offsets = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
        ], dtype="float32")

        corners = box_min + corner_offsets * (box_max - box_min)

        # For each plane, check if all corners are outside that plane
        for plane in self.planes:
            distances = np.dot(corners, plane[:3]) + plane[3]

            if np.all(distances < 0):
                return False

        return True # The box is at least partially inside the frustum

    def is_chunk_in_frustum(self, chunk_coords, chunk_size):
        # Checks if the chunk's AABB is in the frustum
        chunk_min = np.array(chunk_coords, dtype="float32") * chunk_size * block_scale_factor
        chunk_max = chunk_min + (chunk_size * block_scale_factor)

        # Use the same logic as in the "is_box_in_frustum" method
        return self.is_box_in_frustum(chunk_min, chunk_max)
    
def get_chunk_coords(position):
    # Returns the chunk coordinates (as integers) for a given block position
    return (
        int(position[0] // (CHUNK_SIZE * block_scale_factor)),
        int(position[1] // (CHUNK_SIZE * block_scale_factor)),
        int(position[2] // (CHUNK_SIZE * block_scale_factor))
    )

def organize_blocks_into_chunks(blocks):
    # Organizes blocks into a dictionary of chunks
    chunks = {}

    for block in blocks:
        chunk_coords = get_chunk_coords(block.position)
        if chunk_coords not in chunks:
            chunks[chunk_coords] = []
        chunks[chunk_coords].append(block)

    return chunks

# Pygame and OpenGL display initialization
def init_pygame(width, height):
    pygame.init()
    pygame.display.set_caption("PyCraft 0.2")
    icon_path = resource_path("icon.png")
    icon = pygame.image.load(icon_path)
    pygame.display.set_icon(icon)
    pygame.display.set_mode((width, height),DOUBLEBUF | OPENGL)
    pygame.mouse.set_visible(False) # Hide the mouse cursor
    pygame.event.set_grab(True) # Lock the mouse to the window
    glEnable(GL_DEPTH_TEST)
    glDepthMask(GL_TRUE)
    glEnable(GL_BLEND)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

# Perspective and viewport
def init_opengl(width, height):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (width / height), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)

vertices = np.array([
# Back face
-0.5 * block_scale_factor, -0.5 * block_scale_factor, -0.5 * block_scale_factor, 0.0, 0.0, # Bottom-left
    0.5 * block_scale_factor,  0.5 * block_scale_factor, -0.5 * block_scale_factor, 1.0, 1.0, # top-right
    0.5 * block_scale_factor, -0.5 * block_scale_factor, -0.5 * block_scale_factor, 1.0, 0.0, # bottom-right         
    0.5 * block_scale_factor,  0.5 * block_scale_factor, -0.5 * block_scale_factor, 1.0, 1.0, # top-right
-0.5 * block_scale_factor, -0.5 * block_scale_factor, -0.5 * block_scale_factor, 0.0, 0.0, # bottom-left
-0.5 * block_scale_factor,  0.5 * block_scale_factor, -0.5 * block_scale_factor, 0.0, 1.0, # top-left
# Front face
-0.5 * block_scale_factor, -0.5 * block_scale_factor,  0.5 * block_scale_factor, 0.0, 0.0, # bottom-left
    0.5 * block_scale_factor, -0.5 * block_scale_factor,  0.5 * block_scale_factor, 1.0, 0.0, # bottom-right
    0.5 * block_scale_factor,  0.5 * block_scale_factor,  0.5 * block_scale_factor, 1.0, 1.0, # top-right
    0.5 * block_scale_factor,  0.5 * block_scale_factor,  0.5 * block_scale_factor, 1.0, 1.0, # top-right
-0.5 * block_scale_factor,  0.5 * block_scale_factor,  0.5 * block_scale_factor, 0.0, 1.0, # top-left
-0.5 * block_scale_factor, -0.5 * block_scale_factor,  0.5 * block_scale_factor, 0.0, 0.0, # bottom-left
# Left face
-0.5 * block_scale_factor,  0.5 * block_scale_factor,  0.5 * block_scale_factor, 1.0, 0.0, # top-right
-0.5 * block_scale_factor,  0.5 * block_scale_factor, -0.5 * block_scale_factor, 1.0, 1.0, # top-left
-0.5 * block_scale_factor, -0.5 * block_scale_factor, -0.5 * block_scale_factor, 0.0, 1.0, # bottom-left
-0.5 * block_scale_factor, -0.5 * block_scale_factor, -0.5 * block_scale_factor, 0.0, 1.0, # bottom-left
-0.5 * block_scale_factor, -0.5 * block_scale_factor,  0.5 * block_scale_factor, 0.0, 0.0, # bottom-right
-0.5 * block_scale_factor,  0.5 * block_scale_factor,  0.5 * block_scale_factor, 1.0, 0.0, # top-right
# Right face
    0.5 * block_scale_factor,  0.5 * block_scale_factor,  0.5 * block_scale_factor, 1.0, 0.0, # top-left
    0.5 * block_scale_factor, -0.5 * block_scale_factor, -0.5 * block_scale_factor, 0.0, 1.0, # bottom-right
    0.5 * block_scale_factor,  0.5 * block_scale_factor, -0.5 * block_scale_factor, 1.0, 1.0, # top-right         
    0.5 * block_scale_factor, -0.5 * block_scale_factor, -0.5 * block_scale_factor, 0.0, 1.0, # bottom-right
    0.5 * block_scale_factor,  0.5 * block_scale_factor,  0.5 * block_scale_factor, 1.0, 0.0, # top-left
    0.5 * block_scale_factor, -0.5 * block_scale_factor,  0.5 * block_scale_factor, 0.0, 0.0, # bottom-left     
# Bottom face
-0.5 * block_scale_factor, -0.5 * block_scale_factor, -0.5 * block_scale_factor, 0.0, 1.0, # top-right
    0.5 * block_scale_factor, -0.5 * block_scale_factor, -0.5 * block_scale_factor, 1.0, 1.0, # top-left
    0.5 * block_scale_factor, -0.5 * block_scale_factor,  0.5 * block_scale_factor, 1.0, 0.0, # bottom-left
    0.5 * block_scale_factor, -0.5 * block_scale_factor,  0.5 * block_scale_factor, 1.0, 0.0, # bottom-left
-0.5 * block_scale_factor, -0.5 * block_scale_factor,  0.5 * block_scale_factor, 0.0, 0.0, # bottom-right
-0.5 * block_scale_factor, -0.5 * block_scale_factor, -0.5 * block_scale_factor, 0.0, 1.0, # top-right
# Top face
-0.5 * block_scale_factor,  0.5 * block_scale_factor, -0.5 * block_scale_factor, 0.0, 1.0, # top-left
    0.5 * block_scale_factor,  0.5 * block_scale_factor,  0.5 * block_scale_factor, 1.0, 0.0, # bottom-right
    0.5 * block_scale_factor,  0.5 * block_scale_factor, -0.5 * block_scale_factor, 1.0, 1.0, # top-right     
    0.5 * block_scale_factor,  0.5 * block_scale_factor,  0.5 * block_scale_factor, 1.0, 0.0, # bottom-right
-0.5 * block_scale_factor,  0.5 * block_scale_factor, -0.5 * block_scale_factor, 0.0, 1.0, # top-left
-0.5 * block_scale_factor,  0.5 * block_scale_factor,  0.5 * block_scale_factor, 0.0, 0.0 # bottom-left  
], dtype='float32')

vbo_vertices = None
vbo_instance = None
vbo_instance_size = 0

outline_vertices = np.array([
    -0.5, -0.5, -0.5,   0.5, 0.5, -0.5,  # Edge 0-1
    0.5, -0.5, -0.5,    0.5,  0.5, -0.5,  # Edge 1-2
    -0.5,  -0.5, -0.5,   -0.5,  0.5, -0.5,  # Edge 2-3
   -0.5,  -0.5, 0.5,    0.5, -0.5, 0.5,  # Edge 3-0

    # Top face edges (4 edges)
   0.5, 0.5,  0.5,   0.5, 0.5,  0.5,  # Edge 4-5
    -0.5, 0.5,  0.5,    -0.5,  -0.5,  0.5,  # Edge 5-6
    -0.5,  0.5,  0.5,   -0.5,  0.5,  -0.5,  # Edge 6-7
   -0.5,  -0.5,  0.5,   -0.5, -0.5,  -0.5,  # Edge 7-4

    # Vertical edges (4 edges)
   -0.5, -0.5, 0.5,   -0.5, 0.5,  0.5,  # Edge 0-4
    0.5, 0.5, 0.5,    0.5, -0.5,  -0.5,  # Edge 1-5
    0.5,  0.5, -0.5,    0.5,  0.5,  0.5,  # Edge 2-6
   -0.5,  0.5, -0.5,   -0.5,  0.5,  0.5   # Edge 3-7
], dtype='float32')

# Create a Vertex Buffer Object for the cubes
def create_cube_vbo(instance_data, block_scale_factor, force_update=False):
    global vertices, vbo_vertices, vbo_instance, vbo_instance_size

    if vbo_vertices is None or force_update:
        vbo_vertices = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    if vbo_instance is None or force_update:
        vbo_instance = glGenBuffers(1)

    instance_count = len(instance_data)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_instance)

    if instance_data.nbytes != vbo_instance_size or force_update:
        glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_DYNAMIC_DRAW)
        vbo_instance_size = instance_data.nbytes
    else:
        glBufferSubData(GL_ARRAY_BUFFER, 0, instance_data.nbytes, instance_data)

    outline_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, outline_vbo)
    glBufferData(GL_ARRAY_BUFFER, outline_vertices.nbytes, outline_vertices, GL_STATIC_DRAW)

    return vbo_vertices, len(vertices), vbo_instance, instance_count, instance_data, outline_vbo

cached_text_surface = None
cached_text = ""

def drawText(x, y, text):
    global cached_text_surface, cached_text
    if text != cached_text:
        font = pygame.font.SysFont('arial', 28)                                                
        cached_text_surface = font.render(text, True, (0, 0, 0, 255)).convert_alpha()
        cached_text = text
    textData = pygame.image.tostring(cached_text_surface, "RGBA", True)
    glWindowPos2d(x, y)
    glDrawPixels(cached_text_surface.get_width(), cached_text_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)

def compile_shader(shader_source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, shader_source)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) == GL_FALSE:
        print("Shader compilation error: ", glGetShaderInfoLog(shader))
    return shader

def create_shader_program(vertex_shader_path, fragment_shader_path):
    vertex_src = load_shader_source(vertex_shader_path)
    fragment_src = load_shader_source(fragment_shader_path)
    vertex_shader = compile_shader(vertex_src, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_src, GL_FRAGMENT_SHADER)

    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)

    if not glGetProgramiv(program, GL_LINK_STATUS):
        error_log = glGetProgramInfoLog(program)
        print(f"Shader program linking failed: {error_log.decode()}")
        return None

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return program

def load_shader_source(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def draw_cube_outline(cube_position, cube_size, shader_program, outline_vbo):
    # Update the model matrix for this specific cube
    model_matrix = glm.mat4(1.0)
    model_matrix = glm.translate(model_matrix, glm.vec3(cube_position))
    model_matrix = glm.scale(model_matrix, glm.vec3(cube_size))
    glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, glm.value_ptr(model_matrix))

    # Save current opengl states
    polygon_mode = glGetIntegerv(GL_POLYGON_MODE)[0] # Extract the scalar value

    # Configure for outline rendering
    glDepthFunc(GL_LEQUAL)
    glEnable(GL_POLYGON_OFFSET_LINE)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    glLineWidth(2.0) # Line width for the outline
    glDisable(GL_CULL_FACE)

    # Bind the VBO containing the cube vertices
    glBindBuffer(GL_ARRAY_BUFFER, outline_vbo)

    # Enable vertex attributes
    glEnableVertexAttribArray(0) # Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))

    # Draw the cube
    glDrawArrays(GL_LINES, 0, 24)

    # Disable attributes and unbind the buffer
    glDisableVertexAttribArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    # Reset polygon mode
    glPolygonMode(GL_FRONT_AND_BACK, polygon_mode)
    glDisable(GL_POLYGON_OFFSET_LINE)
    glEnable(GL_CULL_FACE)

def get_block_under_cursor(cameraPos, cameraFront, instance_data, half_block_size, max_distance=6.0):
    if instance_data.size == 0:
        return None
    
    # Numpy arrays increase efficiency
    cameraPos = np.array(cameraPos)
    cameraFront = np.array(cameraFront)

    # Computing min and max of blocks in one go
    block_min = instance_data[:, :3] - half_block_size
    block_max = instance_data[: , :3] + half_block_size

    # Vectorized ray intersection test
    inv_cameraFront = 1.0 / cameraFront
    t_min = (block_min - cameraPos) * inv_cameraFront
    t_max = (block_max - cameraPos) * inv_cameraFront
    
    # Use np.where to avoid dividing by 0
    t1 = np.where(cameraFront != 0, np.minimum(t_min, t_max), np.inf)
    t2 = np.where(cameraFront != 0, np.maximum(t_min, t_max), -np.inf)

    t_near = np.max(t1, axis=1) # Maximum entry point
    t_far = np.min(t2, axis=1) # Minimum exit point

    # Valid hits: t_near <= t_far and t_far >= 0
    valid_hits = (t_near <= t_far) & (t_far >= 0)

    # Early exit if no valid hits
    if not np.any(valid_hits):
        return None
    
    # Get the block positions and distances
    valid_blocks = instance_data[valid_hits][:, :3]
    distances = np.linalg.norm(valid_blocks - cameraPos, axis=1) # Distance from camera to block
        
    # Find blocks within the max distance
    within_proximity = distances <= max_distance

    if np.any(within_proximity):
        # Get the closest valid block within proximity
        closest_idx = np.argmin(distances[within_proximity])
        closest_block = valid_blocks[within_proximity][closest_idx]
        return closest_block
    
    return None

def create_block_map(grid_size, grid_height, textures, block_scale_factor):
    chunks = {}
    block_id = 0

    # Define colors for block types
    color_grass = np.array([0.027, 0.878, 0.184, 1.0], dtype="float32") # Green (grass)
    color_cobblestone = np.array([0.259, 0.29, 0.267, 1.0], dtype="float32") # Grey (cobblestone)

    for x in range(grid_size):
        for z in range(grid_size):
            # Random ground level for cobblestone
            cobblestone_height = 90
            grass_height = cobblestone_height + 5
            for y in range(grid_height):
                position = glm.vec3(x * block_scale_factor, y * block_scale_factor, z * block_scale_factor)

                # Determining block type and placement
                if y <= cobblestone_height:
                    # Randomly skipping blocks to create tunnel-like gaps
                    if random.random() > 0.8: # Probability for larger smaller gaps
                        block_type = 1
                        color = color_cobblestone
                        texture = textures[1]
                    else:
                        continue # Skip this block to create a gap
                elif y <= grass_height:
                    if random.random() > 0.5: # Probability of sparser or denser grass
                        block_type = 0
                        color = color_grass
                        texture = textures[0]
                    else:
                        continue # No grass
                else:
                    continue # No blocks above ground level

                block = Block(block_id, block_type, color, position, texture)
                block_id += 1

                # Determine which chunk the block belongs to
                chunk_coords = get_chunk_coords(position)
                if chunk_coords not in chunks:
                    chunks[chunk_coords] = []
                chunks[chunk_coords].append(block)

    return chunks

def blocks_to_instance_data(blocks):
    # Return empty data if blocks are not available
    if not blocks:
        return np.array([], dtype="float32"), 0
    
    instance_count = len(blocks)
    instance_data = np.zeros((instance_count, 9), dtype="float32")

    block_positions = np.array([block.position for block in blocks], dtype="float32")
    block_colors = np.array([block.color for block in blocks], dtype="float32")
    block_textures = np.array([block.texture for block in blocks], dtype="float32")

    instance_data[:, :3] = block_positions
    instance_data[:, 3] = 1.0
    instance_data[:, 4:8] = block_colors
    instance_data[:, 8] = block_textures

    return instance_data, instance_count

def load_texture(file_path):
    # Load the image file using Pillow
    img = Image.open(file_path)
    img = img.transpose(Image.FLIP_TOP_BOTTOM) # Flip the image vertically
    img_data = img.convert("RGBA").tobytes() # Convert image to RGBA format

    # Generate a texture id and bind it
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)

    # Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

    # Upload the texture data to the GPU
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)

    # Generate mipmaps for better scaling
    glGenerateMipmap(GL_TEXTURE_2D)

    # Unbind the texture to avoid modifying it accidentally
    glBindTexture(GL_TEXTURE_2D, 0)

    return texture

# Global cache variables to track attribute configuration state
initialized_attributes = False

def draw_textured_cube(vbo_vertices, vbo_instance, vertex_count, instance_count):
    global initialized_attributes

    # Enable and configure attributes only if not already initialized
    #if not initialized_attributes:
    glEnableVertexAttribArray(0)  # Position attribute
    glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))

    glEnableVertexAttribArray(1)  # Texture coordinates
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(3 * 4))

    glEnableVertexAttribArray(2)  # Instance position attribute
    glBindBuffer(GL_ARRAY_BUFFER, vbo_instance)
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 9 * 4, None)
    glVertexAttribDivisor(2, 1)

    glEnableVertexAttribArray(3)  # Instance color attribute
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 9 * 4, ctypes.c_void_p(4 * 4))
    glVertexAttribDivisor(3, 1)

    glEnableVertexAttribArray(4)  # Texture index attribute
    glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, 9 * 4, ctypes.c_void_p(8 * 4))
    glVertexAttribDivisor(4, 1)

        #initialized_attributes = True  # Mark attributes as initialized

    # Draw the cube instances
    glDrawArraysInstanced(GL_TRIANGLES, 0, vertex_count // 3, instance_count)

    glDisableVertexAttribArray(0)
    glDisableVertexAttribArray(1)
    glDisableVertexAttribArray(2)
    glDisableVertexAttribArray(3)
    glDisableVertexAttribArray(4)

def draw_shadow_cubes(vbo_vertices, vbo_instance, vertex_count, instance_count):
    # Enable and configure only the necessary attribute for shadow mapping
    glEnableVertexAttribArray(0)  # Position attribute
    glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))  # Only position, ignore texture coordinates

    glEnableVertexAttribArray(2)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_instance)
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 9 * 4, ctypes.c_void_p(0))
    glVertexAttribDivisor(2, 1)

    glEnableVertexAttribArray(5)
    glVertexAttribPointer(5, 3, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p((3 + 2) * 4))

    # Draw the cube instances
    if instance_count > 1:
        glDrawArraysInstanced(GL_TRIANGLES, 0, vertex_count // 3, instance_count)
    else:
        glDrawArrays(GL_TRIANGLES, 0, vertex_count // 3)

    # Disable the position attribute to clean up
    glDisableVertexAttribArray(0)
    glDisableVertexAttribArray(2)
    glDisableVertexAttribArray(5)


def handle_collision(player, blocks, block_scale_factor, deltaTime, blockDestroyed):
    # Calculate player's bounding box
    next_position = player.position + player.velocity * deltaTime
    player_min = next_position - glm.vec3(player.width / 2, player.height / 2, player.width / 2)
    player_max = next_position + glm.vec3(player.width / 2, (player.height / 2) - 2, player.width / 2)

    if blockDestroyed:
        player.is_grounded = True
    else:
        player.is_grounded = False

    epsilon = 1e-6

    for block in blocks:
        block_min = block.position - glm.vec3(block_scale_factor / 2)
        block_max = block.position + glm.vec3(block_scale_factor / 2)

        if (player_max.x > block_min.x + epsilon and player_min.x < block_max.x - epsilon and player_max.y > block_min.y + epsilon and player_min.y < block_max.y - epsilon and player_max.z > block_min.z + epsilon and player_min.z < block_max.z - epsilon):
            x_overlap = min(player_max.x - block_min.x, block_max.x - player_min.x)
            y_overlap = min(player_max.y - block_min.y, block_max.y - player_min.y)
            z_overlap = min(player_max.z - block_min.z, block_max.z - player_min.z)

            if y_overlap < x_overlap and y_overlap < z_overlap:
                if player.position[1] < block.position[1]: # Collision from bottom
                    player.velocity[1] = 0
                else: # Collision from top
                    player.position[1] = block_max.y + player.height / 2
                    player.is_grounded = True
                    player.velocity[1] = 0

            elif x_overlap < z_overlap:
                if player.position[0] < block.position[0]:
                    player.position[0] = block_min.x - player.width / 2
                else:
                    player.position[0] = block_max.x + player.width / 2
                player.velocity[0] = 0

            else:
                if player.position[2] < block.position[2]:
                    player.position[2] = block_min.z - player.width / 2
                else:
                    player.position[2] = block_max.z + player.width / 2
                player.velocity[2] = 0

def handle_player_movement(player, keys, cameraFront, cameraUp, cameraSpeed):
    player.velocity[0] = 0
    player.velocity[2] = 0

    if keys[pygame.K_w]:
        player.velocity += cameraSpeed * glm.normalize(glm.vec3(cameraFront.x, 0, cameraFront.z))
    if keys[pygame.K_s]:
        player.velocity -= cameraSpeed * glm.normalize(glm.vec3(cameraFront.x, 0, cameraFront.z))

    if keys[pygame.K_a]:
        player.velocity -= glm.normalize(glm.cross(cameraFront, cameraUp)) * cameraSpeed
    if keys[pygame.K_d]:
        player.velocity += glm.normalize(glm.cross(cameraFront, cameraUp)) * cameraSpeed

    if keys[pygame.K_SPACE]:
        player.jump()

def intersect_planes(p1, p2, p3, tolerance=1e-6):
    """ Find the intersection point of three planes, which will give us a corner of the frustum """
    A = np.array([p1[:3], p2[:3], p3[:3]])  # Normals of the planes (A, B, C)
    b = -np.array([p1[3], p2[3], p3[3]])  # Distance from origin (D)

    # Solve for the intersection point using linear algebra (Ax = b)
    det = np.linalg.det(A)
    if abs(det) <= tolerance:
        print(f"Planes do not intersect properly: det = {det}")
        print(f"Plane 1: {p1}")
        print(f"Plane 2: {p2}")
        print(f"Plane 3: {p3}")
    if abs(det) > tolerance:
        return np.linalg.solve(A, b)
    return None

cached_instance_data = None
cached_instance_count = 0
blocks_changed = True

def get_instance_data(blocks):
    global cached_instance_data, cached_instance_count, blocks_changed
    if blocks_changed:
        cached_instance_data, cached_instance_count = blocks_to_instance_data(blocks)
        blocks_changed = False
    return cached_instance_data, cached_instance_count

changed_chunks = set()

def mark_chunks_changed(chunk_coords):
    changed_chunks.add(chunk_coords)

chunk_instance_data = {}

def update_instance_data_for_changed_chunks(chunks):
    global changed_chunks, chunk_instance_data
    for chunk_coords in visible_chunks:
        chunk_blocks = chunks.get(chunk_coords, [])
        if chunk_coords not in chunk_instance_data or chunk_coords in changed_chunks:
            instance_data, instance_count = blocks_to_instance_data(chunk_blocks)
            chunk_instance_data[chunk_coords] = (instance_data, instance_count)
    changed_chunks.clear()

visible_chunks = set()

def update_visible_chunks(chunks, frustum):
    global visible_chunks
    new_visible_chunks = set()

    for chunk_coords, chunk_blocks in chunks.items():
        if frustum.is_chunk_in_frustum(chunk_coords, CHUNK_SIZE):
            new_visible_chunks.add(chunk_coords)

    entered_chunks = new_visible_chunks - visible_chunks
    exited_chunks = visible_chunks - new_visible_chunks

    visible_chunks = new_visible_chunks

    return entered_chunks, exited_chunks

def update_visible_blocks(chunks, frustum):
    global visible_blocks, visible_chunks
    entered_chunks, exited_chunks = update_visible_chunks(chunks, frustum)

    if entered_chunks or changed_chunks:
        update_instance_data_for_changed_chunks(chunks)

    instance_data_list = [
        chunk_instance_data[chunk_coords][0]
        for chunk_coords in visible_chunks
        if chunk_coords in chunk_instance_data and chunk_instance_data[chunk_coords][0].size > 0
    ]

    instance_counts = [
        chunk_instance_data[chunk_coords][1]
        for chunk_coords in visible_chunks
        if chunk_coords in chunk_instance_data
    ]

    if instance_data_list:
        try:
            instance_data = np.vstack(instance_data_list)
        except ValueError as e:
            print("Error in vstack:", e)
            instance_data = np.empty((0, 9), dtype="float32")
        instance_count = sum(instance_counts)
    else:
        instance_data = np.empty((0, 9), dtype="float32")
        instance_count = 0

    return instance_data, instance_count

last_bound_texture0 = None
last_bound_texture1 = None
last_model_matrix = None
last_view_matrix = None
last_proj_matrix = None

def set_shader_uniforms(shader_program, model, view, proj, grass_texture, cobblestone_texture):
    global last_bound_texture0, last_bound_texture1
    global last_model_matrix, last_view_matrix, last_proj_matrix

    if last_bound_texture0 != grass_texture:
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, grass_texture)
        glUniform1i(glGetUniformLocation(shader_program, "texSampler[0]"), 0)
        last_bound_texture0 = grass_texture

    if last_bound_texture1 != cobblestone_texture:
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, cobblestone_texture)
        glUniform1i(glGetUniformLocation(shader_program, "texSampler[1]"), 1)
        last_bound_texture1 = cobblestone_texture

    if not np.array_equal(model, last_model_matrix):
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, glm.value_ptr(model))
        last_model_matrix = np.copy(model)

    if not np.array_equal(view, last_view_matrix):
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "view"), 1, GL_FALSE, glm.value_ptr(view))
        last_view_matrix = np.copy(view)

    if not np.array_equal(proj, last_proj_matrix):
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "projection"), 1, GL_FALSE, glm.value_ptr(proj))
        last_proj_matrix = np.copy(proj)

def get_block_face_from_intersection(cameraPos, direction, block_position, block_scale_factor):
    """
    Determines the closest face of a block based on a ray from the camera.
    """
    half_size = block_scale_factor / 2
    min_dist = float('inf')
    face = None

    # Define boundaries for each face of the block relative to the block position
    faces = {
        'top':    np.array([0, 1, 0]),
        'bottom': np.array([0, -1, 0]),
        'left':   np.array([-1, 0, 0]),
        'right':  np.array([1, 0, 0]),
        'front':  np.array([0, 0, 1]),
        'back':   np.array([0, 0, -1])
    }

    # Calculate the intersection point with each face and find the closest hit
    for face_name, normal in faces.items():
        # Position of each face (adjusted by half_size in the direction of the normal)
        plane_pos = block_position + normal * half_size
        denom = np.dot(direction, normal)

        # Check if the ray is parallel to the plane (denom close to zero)
        if abs(denom) > 1e-6:
            t = np.dot(plane_pos - cameraPos, normal) / denom

            # If t is positive and closer than previous min_dist, it's a valid intersection
            if t > 0 and t < min_dist:
                intersection_point = cameraPos + direction * t

                # Check if the intersection point lies within the face boundaries
                within_x = block_position[0] - half_size <= intersection_point[0] <= block_position[0] + half_size
                within_y = block_position[1] - half_size <= intersection_point[1] <= block_position[1] + half_size
                within_z = block_position[2] - half_size <= intersection_point[2] <= block_position[2] + half_size

                # Verify that the intersection is within the bounds of the face
                if (
                    (face_name in ('top', 'bottom') and within_x and within_z) or
                    (face_name in ('left', 'right') and within_y and within_z) or
                    (face_name in ('front', 'back') and within_x and within_y)
                ):
                    min_dist = t
                    face = face_name

    return face

def create_depth_map(width, height):
    depthMapFBO = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO)

    depthMap = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, depthMap)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
    borderColor = [1.0, 1.0, 1.0, 1.0]
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor)

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0)
    glDrawBuffer(GL_NONE)
    glReadBuffer(GL_NONE)

    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    return depthMapFBO, depthMap

def render_depth_map(shadow_shader, depthMapFBO, lightSpaceMatrix, model, vbo_vertices, vbo_instance, count, instance_count):
    glUseProgram(shadow_shader)
    glUniformMatrix4fv(glGetUniformLocation(shadow_shader, "lightSpaceMatrix"), 1, GL_FALSE, glm.value_ptr(lightSpaceMatrix))
    glUniformMatrix4fv(glGetUniformLocation(shadow_shader, "model"), 1, GL_FALSE, glm.value_ptr(model))

    glViewport(0, 0, 2048, 2048)
    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO)
    glClear(GL_DEPTH_BUFFER_BIT)
    draw_shadow_cubes(vbo_vertices, vbo_instance, count, instance_count)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glViewport(0, 0, 1152, 648)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glUseProgram(0)

# Main game loop
def game_loop():
    clock = pygame.time.Clock()
    frame_count = 0
    last_time = pygame.time.get_ticks()
    fps = 0
    yaw = -90.0
    pitch = -20.0
    sensitivity = 0.1

    # Time variables
    deltaTime = 0.0
    lastFrame = 0.0

    # Initialize player
    player = Player(position=glm.vec3(5.0, 122.0, 5.0))

    # Camera variables
    cameraPos = glm.vec3(player.position)
    cameraPos[1] += player.height / 2
    cameraUp = glm.vec3(0.0, 1.0, 0.0)
    cameraFront = glm.vec3(0.0, 0.0, -1.0)
    view = glm.mat4()
    view = glm.lookAt(cameraPos, cameraPos + cameraFront, cameraUp)
    direction = glm.vec3()

    # Matrix variables
    proj = glm.mat4()
    proj = glm.perspective(glm.radians(75.0), 1152 / 648, 0.1, 250.0)
    model = glm.mat4(1.0)
    model = glm.scale(model, glm.vec3(1.0))

    light_pos = glm.vec3(0.0, 150.0, 0.0) # Position of the light source
    light_target = glm.vec3(0.0, 0.0, 0.0) # Target of the light source
    lightProjection = glm.ortho(-20.0, 20.0, -20.0, 20.0, 0.1, 150.0)
    lightView = glm.lookAt(light_pos, light_target, glm.vec3(0.0, 0.0, -1.0))
    lightSpaceMatrix = lightProjection * lightView

    # Loading textures
    grass_path = resource_path("grass.png")
    grass_texture = load_texture(grass_path)
    cobblestone_path = resource_path("cobblestone.png")
    cobblestone_texture = load_texture(cobblestone_path)
    textures = [grass_texture, cobblestone_texture]

    # Generate block map
    grid_size = 16
    grid_height = 256
    global block_scale_factor
    chunks = create_block_map(grid_size, grid_height, textures, block_scale_factor)

    depthMapFBO, depthMap = create_depth_map(2048, 2048)
    shadow_shader = create_shader_program(resource_path("shadow_vertex.glsl"), resource_path("shadow_fragment.glsl"))
    shader_program = create_shader_program(resource_path("vertex.glsl"), resource_path("fragment.glsl"))

    global visible_chunks
    visible_chunks = set()
    global chunk_instance_data
    chunk_instance_data = {}
    global changed_chunks
    changed_chunks = set()

    half_block_size = block_scale_factor / 2.0

    block_destroyed = False

    global blocks_changed

    frustum = Frustum(proj, view)
    instance_data, instance_count = update_visible_blocks(chunks, frustum)
    vbo_vertices, count, vbo_instance, instance_count, _, outline_vbo = create_cube_vbo(instance_data, block_scale_factor, force_update=True)

    block_placed = False

    block_destroyed2 = False

    while True:
        keys = pygame.key.get_pressed()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

        # Get mouse movement (relative to the last frame)
        mouse_movement = pygame.mouse.get_rel()
        dx, dy = mouse_movement

        # Update yaw and pitch based on mouse movement
        yaw += dx * sensitivity
        pitch -= dy * sensitivity # Subtract dy to invert mouse Y-axis for typical FPS behavior

        pitch = np.clip(pitch, -89.0, 89.0)

        direction.x = glm.cos(glm.radians(yaw)) * glm.cos(glm.radians(pitch))
        direction.y = glm.sin(glm.radians(pitch))
        direction.z = glm.sin(glm.radians(yaw)) * glm.cos(glm.radians(pitch))
        cameraFront = glm.normalize(direction)

        cameraSpeed = 6.0

        if player.position[1] < 0:
            player.position[0] = 5
            player.position[1] = 122
            player.position[2] = 5

        cameraPos = glm.vec3(player.position)

        view = glm.lookAt(cameraPos, cameraPos + cameraFront, cameraUp)

        # Clear buffers and set the background color to blue
        glClearColor(0.529, 0.808, 0.922, 1) # Light blue color
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT)

        frustum = Frustum(proj, view)
        instance_data, instance_count = update_visible_blocks(chunks, frustum)
        vbo_vertices, count, vbo_instance, instance_count, _, outline_vbo = create_cube_vbo(instance_data, block_scale_factor, force_update=False)

        render_depth_map(shadow_shader, depthMapFBO, lightSpaceMatrix, model, vbo_vertices, vbo_instance, count, instance_count)

        glUseProgram(shader_program)

        if np.any([last_model_matrix != model, last_view_matrix != view, last_proj_matrix != proj]):
            set_shader_uniforms(shader_program, model, view, proj, grass_texture, cobblestone_texture)

        glUniformMatrix4fv(glGetUniformLocation(shader_program, "lightSpaceMatrix"), 1, GL_FALSE, glm.value_ptr(lightSpaceMatrix))
        glUniform3fv(glGetUniformLocation(shader_program, "lightPos"), 1, glm.value_ptr(light_pos))

        # Bind shadow map to texture unit 2
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, depthMap)
        glUniform1i(glGetUniformLocation(shader_program, "shadowMap"), 2)

        if instance_count > 0:
            # Draw visible blocks
            glUniform1i(glGetUniformLocation(shader_program, "isOutline"), GL_FALSE)
            draw_textured_cube(vbo_vertices, vbo_instance, count, instance_count)

        # Cast ray from the camera to detect block under cursor
        block_under_cursor = get_block_under_cursor(cameraPos, cameraFront, instance_data, half_block_size, max_distance=6.0)
        if block_under_cursor is not None:
            glUniform1i(glGetUniformLocation(shader_program, "isOutline"), GL_TRUE)
            draw_cube_outline(block_under_cursor, block_scale_factor, shader_program, outline_vbo)
            glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, glm.value_ptr(model))
            if event.type == pygame.MOUSEBUTTONDOWN:

                if event.button == 1 and not block_destroyed2: # Left mouse button - destroy the block
                    # Remove the block under the cursor
                    chunk_coords = get_chunk_coords(block_under_cursor)
                    if chunk_coords in chunks:
                        chunks[chunk_coords] = [block for block in chunks[chunk_coords] if not np.all(block.position == block_under_cursor)]
                        mark_chunks_changed(chunk_coords)
                    # Update instance data
                    instance_data, instance_count = blocks_to_instance_data([block for blocks in chunks.values() for block in blocks])
                    # Recreate the VBO
                    vbo_vertices, count, vbo_instance, instance_count, _, outline_vbo = create_cube_vbo(instance_data, block_scale_factor, force_update=True)
                    blocks_changed = True
                    block_destroyed = True
                    block_placed = False
                    block_destroyed2 = True

                elif event.button == 3 and not block_placed:  # Right mouse button - place block
                    # Determine which face of the block is being looked at
                    block_face = get_block_face_from_intersection(cameraPos, glm.normalize(cameraFront), block_under_cursor, block_scale_factor)
                    # Initialize new block position as the current block's position
                    new_position = np.array(block_under_cursor)

                    if block_face == 'top':
                        new_position[1] += block_scale_factor
                    elif block_face == 'bottom':
                        new_position[1] -= block_scale_factor
                    elif block_face == 'left':
                        new_position[0] -= block_scale_factor
                    elif block_face == 'right':
                        new_position[0] += block_scale_factor
                    elif block_face == 'front':
                        new_position[2] += block_scale_factor
                    elif block_face == 'back':
                        new_position[2] -= block_scale_factor

                    # Place a new cobblestone block at the calculated position
                    block_to_place = glm.vec3(*new_position)
                    if block_to_place is not None:
                        chunk_coords = get_chunk_coords(block_to_place)
                        color_cobblestone = np.array([0.259, 0.29, 0.267, 1.0], dtype="float32")
                        new_block = Block(block_id=len([block for chunk_blocks in chunks.values() for block in chunk_blocks]), block_type=1, color=color_cobblestone, position=block_to_place, texture=textures[1])
                        if chunk_coords not in chunks:
                            chunks[chunk_coords] = []
                        chunks[chunk_coords].append(new_block)
                        mark_chunks_changed(chunk_coords)

                        all_blocks = [block for chunk_blocks in chunks.values() for block in chunk_blocks]

                        # Update instance data
                        instance_data, instance_count = blocks_to_instance_data(all_blocks)

                        # Recreate the VBO to reflect changes
                        vbo_vertices, count, vbo_instance, instance_count, _, outline_vbo = create_cube_vbo(instance_data, block_scale_factor, force_update=True)
                        blocks_changed = True
                        block_placed = True
            
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    block_destroyed2 = False
                if event.button == 3:
                    block_placed = False

        glUseProgram(0)

        player.apply_gravity(deltaTime)
        visible_blocks = [block for chunk_coords in visible_chunks for block in chunks.get(chunk_coords, [])]
        handle_collision(player, visible_blocks, block_scale_factor, deltaTime, block_destroyed)
        player.update_position(deltaTime)

        handle_player_movement(player, keys, cameraFront, cameraUp, cameraSpeed)

        block_destroyed = False

        # Clock for speed normalization
        currentFrame = time.time()
        deltaTime = currentFrame - lastFrame
        lastFrame = currentFrame

        # Calculate and display FPS
        frame_count += 1
        if pygame.time.get_ticks() - last_time >= 500:  # Every half second
            fps = frame_count
            frame_count = 0
            last_time = pygame.time.get_ticks()

        drawText(1000, 600, f'FPS: {fps}')

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(120)

# Initialize the game
def main():
    width, height = 1152, 648
    init_pygame(width, height)
    init_opengl(width, height)
    game_loop()
    #cProfile.run("game_loop()", filename="game_profile.prof")

if __name__ == "__main__":
    main()