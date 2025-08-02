use opencv::{
    core,
    highgui,
    imgcodecs,
    imgproc,
    prelude::*,
    Result,
};

fn main() -> Result<()> {
    println!("OpenCV Rust Image Operations");
    
    // Load an image
    let img = imgcodecs::imread("input.jpg", imgcodecs::IMREAD_COLOR)?;
    
    if img.empty() {
        println!("Could not load image");
        return Ok(());
    }
    
    println!("Image loaded: {}x{}", img.cols(), img.rows());
    
    // Convert to grayscale
    let mut gray = Mat::default();
    imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
    
    // Display the image (optional)
    highgui::imshow("Original", &img)?;
    highgui::imshow("Grayscale", &gray)?;
    highgui::wait_key(0)?;
    
    Ok(())
}
