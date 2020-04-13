#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        use crate::*;
        use glfw::{ClientApiHint, WindowHint};

        use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
        use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
        use vulkano::device::{Device, DeviceExtensions};
        use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, Subpass, RenderPassAbstract};
        use vulkano::image::SwapchainImage;
        use vulkano::instance::{Instance, PhysicalDevice};
        use vulkano::pipeline::GraphicsPipeline;
        use vulkano::pipeline::viewport::Viewport;
        use vulkano::swapchain::{AcquireError, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError, ColorSpace, FullscreenExclusive};
        use vulkano::swapchain;
        use vulkano::sync::{GpuFuture, FlushError};
        use vulkano::sync;

        use std::sync::Arc;

        let mut cv = init();

        let mut glfw = &mut cv.glfw;

        glfw.window_hint(WindowHint::ClientApi(ClientApiHint::NoApi));
        let (mut window, events) = glfw
            .create_window(300, 300, "Hello this is window", glfw::WindowMode::Windowed)
            .expect("Failed to create GLFW window.");


        /// This method is called once during initialization, then again whenever the window is resized
        fn window_size_dependent_setup(
            images: &[Arc<SwapchainImage<()>>],
            render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
            dynamic_state: &mut DynamicState
        ) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
            let dimensions = images[0].dimensions();

            let viewport = Viewport {
                origin: [0.0, 0.0],
                dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                depth_range: 0.0 .. 1.0,
            };
            dynamic_state.viewports = Some(vec!(viewport));

            images.iter().map(|image| {
                Arc::new(
                    Framebuffer::start(render_pass.clone())
                        .add(image.clone()).unwrap()
                        .build().unwrap()
                ) as Arc<dyn FramebufferAbstract + Send + Sync>
            }).collect::<Vec<_>>()
        }

        // The first step of any Vulkan program is to create an instance.
        //
        // When we create an instance, we have to pass a list of extensions that we want to enable.
        //
        // All the window-drawing functionalities are part of non-core extensions that we need
        // to enable manually. To do so, we ask the `vulkano_win` crate for the list of extensions
        // required to draw to a window.

        // Now creating the instance.
        let instance = cv.vulkano_instance.clone();

        // We then choose which physical device to use.
        //
        // In a real application, there are three things to take into consideration:
        //
        // - Some devices may not support some of the optional features that may be required by your
        //   application. You should filter out the devices that don't support your app.
        //
        // - Not all devices can draw to a certain surface. Once you create your window, you have to
        //   choose a device that is capable of drawing to it.
        //
        // - You probably want to leave the choice between the remaining devices to the user.
        //
        // For the sake of the example we are just going to use the first device, which should work
        // most of the time.
        let physical = PhysicalDevice::enumerate(&instance).map(|x| {println!("{}",x.name());
                                                                     x
        }).next().unwrap();

        // Some little debug infos.
        println!(
            "Using device: {} (type: {:?})",
            physical.name(),
            physical.ty()
        );

        // The objective of this example is to draw a triangle on a window. To do so, we first need to
        // create the window.
        //
        // This is done by creating a `WindowBuilder` from the `winit` crate, then calling the
        // `build_vk_surface` method provided by the `VkSurfaceBuild` trait from `vulkano_win`. If you
        // ever get an error about `build_vk_surface` being undefined in one of your projects, this
        // probably means that you forgot to import this trait.
        //
        // This returns a `vulkano::swapchain::Surface` object that contains both a cross-platform winit
        // window and a cross-platform Vulkan surface that represents the surface of the window.
        let mut vksurf: vk_sys::SurfaceKHR = 0;

        assert_eq!(window.create_window_surface(cv.instance,std::ptr::null(),&mut vksurf),vk_sys::SUCCESS);

        let surface =
            Arc::new(unsafe{vulkano::swapchain::Surface::<()>::from_raw_surface(instance.clone(), vksurf, ())});

        // The next step is to choose which GPU queue will execute our draw commands.
        //
        // Devices can provide multiple queues to run commands in parallel (for example a draw queue
        // and a compute queue), similar to CPU threads. This is something you have to have to manage
        // manually in Vulkan.
        //
        // In a real-life application, we would probably use at least a graphics queue and a transfers
        // queue to handle data transfers in parallel. In this example we only use one queue.
        //
        // We have to choose which queues to use early on, because we will need this info very soon.

        let queue_family = physical
            .queue_families()
            .find(|&q| {
                // We take the first queue that supports drawing to our window.
                q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
            })
            .unwrap();

        // Now initializing the device. This is probably the most important object of Vulkan.
        //
        // We have to pass five parameters when creating a device:
        //
        // - Which physical device to connect to.
        //
        // - A list of optional features and extensions that our program needs to work correctly.
        //   Some parts of the Vulkan specs are optional and must be enabled manually at device
        //   creation. In this example the only thing we are going to need is the `khr_swapchain`
        //   extension that allows us to draw to a window.
        //
        // - A list of layers to enable. This is very niche, and you will usually pass `None`.
        //
        // - The list of queues that we are going to use. The exact parameter is an iterator whose
        //   items are `(Queue, f32)` where the floating-point represents the priority of the queue
        //   between 0.0 and 1.0. The priority of the queue is a hint to the implementation about how
        //   much it should prioritize queues between one another.
        //
        // The list of created queues is returned by the function alongside with the device.
        let device_ext = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::none()
        };
        let (device, mut queues) = Device::new(
            physical,
            physical.supported_features(),
            &device_ext,
            [(queue_family, 0.5)].iter().cloned(),
        )
        .unwrap();

        // Since we can request multiple queues, the `queues` variable is in fact an iterator. In this
        // example we use only one queue, so we just retrieve the first and only element of the
        // iterator and throw it away.
        let queue = queues.next().unwrap();

        // Before we can draw on the surface, we have to create what is called a swapchain. Creating
        // a swapchain allocates the color buffers that will contain the image that will ultimately
        // be visible on the screen. These images are returned alongside with the swapchain.
        let (mut swapchain, images) = {
            // Querying the capabilities of the surface. When we create the swapchain we can only
            // pass values that are allowed by the capabilities.
            let caps = surface.capabilities(physical).unwrap();
            let usage = caps.supported_usage_flags;

            // The alpha mode indicates how the alpha value of the final image will behave. For example
            // you can choose whether the window will be opaque or transparent.
            let alpha = caps.supported_composite_alpha.iter().next().unwrap();

            // Choosing the internal format that the images will have.
            let format = caps.supported_formats[0].0;

            // The dimensions of the window, only used to initially setup the swapchain.
            // NOTE:
            // On some drivers the swapchain dimensions are specified by `caps.current_extent` and the
            // swapchain size must use these dimensions.
            // These dimensions are always the same as the window dimensions
            //
            // However other drivers dont specify a value i.e. `caps.current_extent` is `None`
            // These drivers will allow anything but the only sensible value is the window dimensions.
            //
            // Because for both of these cases, the swapchain needs to be the window dimensions, we just use that.
            let dimensions: [u32; 2] = [window.get_size().0 as u32, window.get_size().1 as u32];

            // Please take a look at the docs for the meaning of the parameters we didn't mention.
            Swapchain::new(
                device.clone(),
                surface.clone(),
                caps.min_image_count,
                format,
                dimensions,
                1,
                usage,
                &queue,
                SurfaceTransform::Identity,
                alpha,
                PresentMode::Fifo,
                FullscreenExclusive::Default,
                true,
                ColorSpace::SrgbNonLinear,
            )
            .unwrap()
        };

        // We now create a buffer that will store the shape of our triangle.
        let vertex_buffer = {
            #[derive(Default, Debug, Clone)]
            struct Vertex {
                position: [f32; 2],
            }
            vulkano::impl_vertex!(Vertex, position);

            CpuAccessibleBuffer::from_iter(
                device.clone(),
                BufferUsage::all(),
                false,
                [
                    Vertex {
                        position: [-0.5, -0.25],
                    },
                    Vertex {
                        position: [0.0, 0.5],
                    },
                    Vertex {
                        position: [0.25, -0.1],
                    },
                ]
                .iter()
                .cloned(),
            )
            .unwrap()
        };

        // The next step is to create the shaders.
        //
        // The raw shader creation API provided by the vulkano library is unsafe, for various reasons.
        //
        // An overview of what the `vulkano_shaders::shader!` macro generates can be found in the
        // `vulkano-shaders` crate docs. You can view them at https://docs.rs/vulkano-shaders/
        //
        // TODO: explain this in details
        mod vs {
            vulkano_shaders::shader! {
                ty: "vertex",
                src: "
				#version 450
				layout(location = 0) in vec2 position;
				void main() {
					gl_Position = vec4(position, 0.0, 1.0);
				}
			"
            }
        }

        mod fs {
            vulkano_shaders::shader! {
                ty: "fragment",
                src: "
				#version 450
				layout(location = 0) out vec4 f_color;
				void main() {
					f_color = vec4(1.0, 0.0, 0.0, 1.0);
				}
			"
            }
        }

        let vs = vs::Shader::load(device.clone()).unwrap();
        let fs = fs::Shader::load(device.clone()).unwrap();

        // At this point, OpenGL initialization would be finished. However in Vulkan it is not. OpenGL
        // implicitly does a lot of computation whenever you draw. In Vulkan, you have to do all this
        // manually.

        // The next step is to create a *render pass*, which is an object that describes where the
        // output of the graphics pipeline will go. It describes the layout of the images
        // where the colors, depth and/or stencil information will be written.
        let render_pass = Arc::new(
            vulkano::single_pass_renderpass!(
                device.clone(),
                attachments: {
                    // `color` is a custom name we give to the first and only attachment.
                    color: {
                        // `load: Clear` means that we ask the GPU to clear the content of this
                        // attachment at the start of the drawing.
                        load: Clear,
                        // `store: Store` means that we ask the GPU to store the output of the draw
                        // in the actual image. We could also ask it to discard the result.
                        store: Store,
                        // `format: <ty>` indicates the type of the format of the image. This has to
                        // be one of the types of the `vulkano::format` module (or alternatively one
                        // of your structs that implements the `FormatDesc` trait). Here we use the
                        // same format as the swapchain.
                        format: swapchain.format(),
                        // TODO:
                        samples: 1,
                    }
                },
                pass: {
                    // We use the attachment named `color` as the one and only color attachment.
                    color: [color],
                    // No depth-stencil attachment is indicated with empty brackets.
                    depth_stencil: {}
                }
            )
            .unwrap(),
        );

        // Before we draw we have to create what is called a pipeline. This is similar to an OpenGL
        // program, but much more specific.
        let pipeline = Arc::new(
            GraphicsPipeline::start()
                // We need to indicate the layout of the vertices.
                // The type `SingleBufferDefinition` actually contains a template parameter corresponding
                // to the type of each vertex. But in this code it is automatically inferred.
                .vertex_input_single_buffer()
                // A Vulkan shader can in theory contain multiple entry points, so we have to specify
                // which one. The `main` word of `main_entry_point` actually corresponds to the name of
                // the entry point.
                .vertex_shader(vs.main_entry_point(), ())
                // The content of the vertex buffer describes a list of triangles.
                .triangle_list()
                // Use a resizable viewport set to draw over the entire window
                .viewports_dynamic_scissors_irrelevant(1)
                // See `vertex_shader`.
                .fragment_shader(fs.main_entry_point(), ())
                // We have to indicate which subpass of which render pass this pipeline is going to be used
                // in. The pipeline will only be usable from this particular subpass.
                .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                // Now that our builder is filled, we call `build()` to obtain an actual pipeline.
                .build(device.clone())
                .unwrap(),
        );

        // Dynamic viewports allow us to recreate just the viewport when the window is resized
        // Otherwise we would have to recreate the whole pipeline.
        let mut dynamic_state = DynamicState {
            line_width: None,
            viewports: None,
            scissors: None,
            compare_mask: None,
            write_mask: None,
            reference: None,
        };

        // The render pass we created above only describes the layout of our framebuffers. Before we
        // can draw we also need to create the actual framebuffers.
        //
        // Since we need to draw to multiple images, we are going to create a different framebuffer for
        // each image.
        let mut framebuffers =
            window_size_dependent_setup(&images, render_pass.clone(), &mut dynamic_state);

        // Initialization is finally finished!

        // In some situations, the swapchain will become invalid by itself. This includes for example
        // when the window is resized (as the images of the swapchain will no longer match the
        // window's) or, on Android, when the application went to the background and goes back to the
        // foreground.
        //
        // In this situation, acquiring a swapchain image or presenting it will return an error.
        // Rendering to an image of that swapchain will not produce any error, but may or may not work.
        // To continue rendering, we need to recreate the swapchain by creating a new swapchain.
        // Here, we remember that we need to do this for the next loop iteration.
        let mut recreate_swapchain = false;

        // In the loop below we are going to submit commands to the GPU. Submitting a command produces
        // an object that implements the `GpuFuture` trait, which holds the resources for as long as
        // they are in use by the GPU.
        //
        // Destroying the `GpuFuture` blocks until the GPU is finished executing it. In order to avoid
        // that, we store the submission of the previous frame here.
        let mut previous_frame_end =
            Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>);

        while !window.should_close() {
            glfw.poll_events();
                    // It is important to call this function from time to time, otherwise resources will keep
                    // accumulating and you will eventually reach an out of memory error.
                    // Calling this function polls various fences in order to determine what the GPU has
                    // already processed, and frees the resources that are no longer needed.
                    previous_frame_end.as_mut().unwrap().cleanup_finished();

                    // Whenever the window resizes we need to recreate everything dependent on the window size.
                    // In this example that includes the swapchain, the framebuffers and the dynamic state viewport.
                    if recreate_swapchain {
                        // Get the new dimensions of the window.
                        let dimensions: [u32; 2] = [window.get_size().0 as u32, window.get_size().1 as u32];
                        
                        let (new_swapchain, new_images) =
                            match swapchain.recreate_with_dimensions(dimensions) {
                                Ok(r) => r,
                                // This error tends to happen when the user is manually resizing the window.
                                // Simply restarting the loop is the easiest way to fix this issue.
                                Err(SwapchainCreationError::UnsupportedDimensions) => continue,
                                Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                            };

                        swapchain = new_swapchain;
                        // Because framebuffers contains an Arc on the old swapchain, we need to
                        // recreate framebuffers as well.
                        framebuffers = window_size_dependent_setup(
                            &new_images,
                            render_pass.clone(),
                            &mut dynamic_state,
                        );
                        recreate_swapchain = false;
                    }

                    // Before we can draw on the output, we have to *acquire* an image from the swapchain. If
                    // no image is available (which happens if you submit draw commands too quickly), then the
                    // function will block.
                    // This operation returns the index of the image that we are allowed to draw upon.
                    //
                    // This function can block if no image is available. The parameter is an optional timeout
                    // after which the function call will return an error.
                    let (image_num, suboptimal, acquire_future) =
                        match swapchain::acquire_next_image(swapchain.clone(), None) {
                            Ok(r) => r,
                            Err(AcquireError::OutOfDate) => {
                                recreate_swapchain = true;
                                continue;
                            }
                            Err(e) => panic!("Failed to acquire next image: {:?}", e),
                        };

                    // acquire_next_image can be successful, but suboptimal. This means that the swapchain image
                    // will still work, but it may not display correctly. With some drivers this can be when
                    // the window resizes, but it may not cause the swapchain to become out of date.
                    if suboptimal {
                        recreate_swapchain = true;
                    }

                    // Specify the color to clear the framebuffer with i.e. blue
                    let clear_values = vec![[0.0, 0.0, 1.0, 1.0].into()];

                    // In order to draw, we have to build a *command buffer*. The command buffer object holds
                    // the list of commands that are going to be executed.
                    //
                    // Building a command buffer is an expensive operation (usually a few hundred
                    // microseconds), but it is known to be a hot path in the driver and is expected to be
                    // optimized.
                    //
                    // Note that we have to pass a queue family when we create the command buffer. The command
                    // buffer will only be executable on that given queue family.
                    let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(
                        device.clone(),
                        queue.family(),
                    )
                    .unwrap()
                    // Before we can draw, we have to *enter a render pass*. There are two methods to do
                    // this: `draw_inline` and `draw_secondary`. The latter is a bit more advanced and is
                    // not covered here.
                    //
                    // The third parameter builds the list of values to clear the attachments with. The API
                    // is similar to the list of attachments when building the framebuffers, except that
                    // only the attachments that use `load: Clear` appear in the list.
                    .begin_render_pass(framebuffers[image_num].clone(), false, clear_values)
                    .unwrap()
                    // We are now inside the first subpass of the render pass. We add a draw command.
                    //
                    // The last two parameters contain the list of resources to pass to the shaders.
                    // Since we used an `EmptyPipeline` object, the objects have to be `()`.
                    .draw(
                        pipeline.clone(),
                        &dynamic_state,
                        vertex_buffer.clone(),
                        (),
                        (),
                    )
                    .unwrap()
                    // We leave the render pass by calling `draw_end`. Note that if we had multiple
                    // subpasses we could have called `next_inline` (or `next_secondary`) to jump to the
                    // next subpass.
                    .end_render_pass()
                    .unwrap()
                    // Finish building the command buffer by calling `build`.
                    .build()
                    .unwrap();

                    let future = previous_frame_end
                        .take()
                        .unwrap()
                        .join(acquire_future)
                        .then_execute(queue.clone(), command_buffer)
                        .unwrap()
                        // The color output is now expected to contain our triangle. But in order to show it on
                        // the screen, we have to *present* the image by calling `present`.
                        //
                        // This function does not actually present the image immediately. Instead it submits a
                        // present command at the end of the queue. This means that it will only be presented once
                        // the GPU has finished executing the command buffer that draws the triangle.
                        .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                        .then_signal_fence_and_flush();

                    match future {
                        Ok(future) => {
                            previous_frame_end = Some(Box::new(future) as Box<_>);
                        }
                        Err(FlushError::OutOfDate) => {
                            recreate_swapchain = true;
                            previous_frame_end =
                                Some(Box::new(sync::now(device.clone())) as Box<_>);
                        }
                        Err(e) => {
                            println!("Failed to flush future: {:?}", e);
                            previous_frame_end =
                                Some(Box::new(sync::now(device.clone())) as Box<_>);
                        }
                    }
                
            }
    }
}

pub struct Window {}

use glfw::{ClientApiHint, WindowHint};

impl Window {
    pub fn new(cv: &mut CrossVulkan) {
        cv.glfw
            .window_hint(WindowHint::ClientApi(ClientApiHint::NoApi));
        let (mut window, events) = cv
            .glfw
            .create_window(300, 300, "Hello this is window", glfw::WindowMode::Windowed)
            .expect("Failed to create GLFW window.");
    }
}

use glfw::Glfw;
use std::ffi::CString;
use std::mem;
use std::os::raw::c_void;
use std::ptr;
use std::sync::Arc;
use vk_sys::{
    self as vk, EntryPoints, Instance as VkInstance, InstanceCreateInfo, InstancePointers,
    Result as VkResult,
};

#[cfg(not(feature = "vulkano-support"))]
pub struct CrossVulkan {
    pub instance: VkInstance,
    glfw: Glfw,
    instance_ptrs: InstancePointers,
}

#[cfg(not(feature = "vulkano-support"))]
pub fn init() -> CrossVulkan {
    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();
    assert!(glfw.vulkan_supported());

    glfw.set_error_callback(glfw::LOG_ERRORS);

    let required_extensions = glfw.get_required_instance_extensions().unwrap_or(vec![]);

    println!("Vulkan required extensions: {:?}", required_extensions);

    let mut entry_points: EntryPoints = EntryPoints::load(|func| {
        glfw.get_instance_proc_address_raw(0, func.to_str().unwrap()) as *const c_void
    });

    let instance: VkInstance = unsafe { create_instance(&mut entry_points, required_extensions) };

    let instance_ptrs: InstancePointers = InstancePointers::load(|func| {
        glfw.get_instance_proc_address_raw(instance, func.to_str().unwrap()) as *const c_void
    });

    CrossVulkan {
        instance,
        glfw,
        instance_ptrs,
    }
}
#[cfg(not(feature = "vulkano-support"))]
pub fn deinit(mut cv: CrossVulkan) {
    unsafe {
        destroy_instance(cv.instance, &mut cv.instance_ptrs);
    }
}

#[cfg(not(feature = "vulkano-support"))]
unsafe fn create_instance(entry_points: &mut EntryPoints, extensions: Vec<String>) -> VkInstance {
    let mut instance: VkInstance = mem::MaybeUninit::uninit().assume_init();

    let cexts: Vec<CString> = extensions
        .clone()
        .into_iter()
        .map(|x| CString::new(x).unwrap())
        .collect();

    let mut ptrs: Vec<&[u8]> = Vec::new();

    for x in &cexts {
        ptrs.push(x.as_bytes());
    }

    let mut ptrs2: Vec<*const std::os::raw::c_char> = Vec::new();

    for x in 0..ptrs.len() {
        ptrs2.push(ptrs[x].as_ptr() as *const i8);
    }

    let ptr = ptrs2.as_ptr() as *const *const std::os::raw::c_char;

    //This is literally the bare minimum required to create a blank instance
    //You'll want to fill in this with real data yourself
    let info: InstanceCreateInfo = InstanceCreateInfo {
        sType: vk::STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pNext: ptr::null(),
        flags: 0,
        pApplicationInfo: ptr::null(),
        enabledLayerCount: 0,
        ppEnabledLayerNames: ptr::null(),
        //These two should use the extensions returned by window.get_required_instance_extensions
        enabledExtensionCount: 2,
        ppEnabledExtensionNames: ptr,
    };

    let res: VkResult = entry_points.CreateInstance(
        &info as *const InstanceCreateInfo,
        ptr::null(),
        &mut instance as *mut VkInstance,
    );

    assert_eq!(res, vk::SUCCESS);

    instance
}
#[cfg(not(feature = "vulkano-support"))]
unsafe fn destroy_instance(instance: VkInstance, instance_ptrs: &mut InstancePointers) {
    instance_ptrs.DestroyInstance(instance, ptr::null());
}

#[cfg(feature = "vulkano-support")]
pub struct CrossVulkan {
    pub instance: VkInstance,
    glfw: Glfw,
    pub vulkano_instance: Arc<vulkano::instance::Instance>,
}

#[cfg(feature = "vulkano-support")]
pub fn init() -> CrossVulkan {
    use smallvec::SmallVec;
    use std::borrow::Cow;
    use std::error;
    use std::ffi::CStr;
    use std::ffi::CString;
    use std::fmt;
    use std::hash::Hash;
    use std::hash::Hasher;
    use std::mem;
    use std::mem::MaybeUninit;
    use std::ops::Deref;
    use std::ptr;
    use std::slice;
    use std::sync::Arc;
    use vulkano::instance::loader;
    use vulkano::instance::loader::FunctionPointers;
    use vulkano::instance::loader::Loader;
    use vulkano::instance::loader::LoadingError;

    // Same as Cow but less annoying.
    enum OwnedOrRef<T: 'static> {
        _Owned(T),
        _Ref(&'static T),
    }

    struct PhysicalDeviceInfos {
        _device: vk::PhysicalDevice,
        _properties: vk::PhysicalDeviceProperties,
        _queue_families: Vec<vk::QueueFamilyProperties>,
        _memory: vk::PhysicalDeviceMemoryProperties,
        //available_features: Features,
    }

    struct FakeVulkanoInstance {
        instance: vk_sys::Instance,
        //alloc: Option<Box<Alloc + Send + Sync>>,
        _physical_devices: Vec<PhysicalDeviceInfos>,
        vk: vk_sys::InstancePointers,
        _extensions: RawInstanceExtensions,
        _layers: SmallVec<[CString; 16]>,
        _function_pointers: OwnedOrRef<FunctionPointers<Box<dyn Loader + Send + Sync>>>,
    }

    use vulkano::instance::{Instance, RawInstanceExtensions};
    use vulkano::VulkanObject;

    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();
    assert!(glfw.vulkan_supported());

    glfw.set_error_callback(glfw::LOG_ERRORS);

    let required_extensions = glfw.get_required_instance_extensions().unwrap_or(vec![]);

    println!("Vulkan required extensions: {:?}", required_extensions);

    let cexts: Vec<CString> = required_extensions
        .into_iter()
        .map(|x| CString::new(x).unwrap())
        .collect();

    let instance = Instance::new(
        None,
        RawInstanceExtensions::new(cexts.into_iter()),
        Vec::new().into_iter(),
    )
    .expect("Could not create vulkano instance");

    let fi: FakeVulkanoInstance = unsafe { std::mem::transmute_copy(instance.as_ref()) };

    let cv = CrossVulkan {
        instance: fi.instance,
        glfw,
        vulkano_instance: instance,
    };

    std::mem::forget(fi);

    cv
}

#[cfg(not(feature = "vulkano-support"))]
pub fn deinit(mut cv: CrossVulkan) {}
