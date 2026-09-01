// set path
rootDir = "C:/Users/Chenxi Zhou/Desktop/VPT/chenxi_segmentation_delivery/masks/";
subFolder = "vptlike_cp2.2.3/";

// test on some sections
embryoList = newArray("Embryo08","Embryo09","Embryo10",
                       "Embryo11","Embryo12","Embryo13");
// can change to whole dataset：
//embryoList = newArray("Embryo04","Embryo05",
//                       "Embryo06","Embryo07","Embryo08","Embryo09","Embryo10",
//                       "Embryo11","Embryo12","Embryo13");

for (e = 0; e < embryoList.length; e++) {
    embryo = embryoList[e];
    baseDir = rootDir + embryo + "/" + subFolder;

    compareFolder(baseDir, "mosaic_Cellbound3_z3_DAPI_mask.tif");
    compareFolder(baseDir + "ecto/", "mosaic_Cellbound3_z3_DAPI_mask_ecto.tif");
    compareFolder(baseDir + "endo/", "mosaic_Cellbound3_z3_DAPI_mask_endo.tif");
}

// save the results
if (isOpen("Mask comparison results")) {
    selectWindow("Mask comparison results");
    saveAs("Results", rootDir + "Mask_comparison_results_summary.csv");
    selectWindow("Log");
    saveAs("Results", rootDir + "MIC-Log.txt");
}

print("Task finished");

// function to conpare
function compareFolder(folder, truthFile) {
    if (!File.exists(folder + truthFile)) {
        print("!! cannot find truth folder，skip the folder: " + folder);
        return;
    }

    // output path
    outDir = folder + "compare_results/";
    if (!File.exists(outDir)) File.makeDirectory(outDir);

    list = getFileList(folder);

    for (i = 0; i < list.length; i++) {
        fname = list[i];

        if (!endsWith(fname, ".tif")) continue;
        if (fname == truthFile) continue;
        if (indexOf(fname, "__VS__") >= 0) continue;   
        if (File.isDirectory(folder + fname)) continue; // skip empty folder

        testFile = fname;
        testBase = replace(testFile, ".tif", "");
        print("比较: " + truthFile + "  VS  " + testFile);

        open(folder + truthFile);
        selectImage(truthFile);
        open(folder + testFile);
        selectImage(testFile);

        run("Mask instant Comparator",
            "truth_mask_image=" + truthFile +
            " test_mask_image=" + testFile +
            " object_(varying_iou) minimum_iou_threshold=0.50 maximum_iou_threshold=1" +
            " increment_of_iou_threshold=0.05 show_graphs show_gt_objects_correspondence_table" +
            " minimum_size_for_objects=0 minimum_distance_to_border=0");

        plotTitle = "plots " + truthFile + "__VS__" + testFile;
        if (isOpen(plotTitle)) {
            selectWindow(plotTitle);
            saveAs("Tiff", outDir + testBase + "_" + plotTitle + ".tif"); // save file
            close();
        } else {
            print("  !! no expected plot window: " + plotTitle);
        }

        if (isOpen("Objects correspondences")) {
            selectWindow("Objects correspondences");
            saveAs("Results", outDir + testBase + "_Objects_correspondences.csv");
            run("Close");
        }

        if (isOpen("Mask comparison Object with IoU thresholds")) {
            selectWindow("Mask comparison Object with IoU thresholds");
            saveAs("Results", outDir + testBase + "_IoU_thresholds.csv");
            run("Close");
        }

        close(truthFile);
        close(testFile);
       //empty ROI manager
        if (isOpen("ROI Manager")) {
            roiManager("reset");
        }
    }
}