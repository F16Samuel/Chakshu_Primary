// src/pages/Register.tsx

import { useState, useRef, useCallback } from "react";
import { Camera, Upload, User, CheckCircle } from "lucide-react";
import Webcam from "react-webcam";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { FileUpload } from "@/components/upload/FileUpload";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { TopButtons } from "@/components/TopButtons"; // Import the TopButtons component

// Use environment variable for the Node.js backend URL
const NODE_BACKEND_URL = import.meta.env.VITE_NODE_BACKEND_URL;

// Define the Node.js proxy endpoint for registration
const REGISTRATION_API_PROXY_URL = `${NODE_BACKEND_URL}/api/registration`;

export default function Register() {
  const { toast } = useToast();
  const [formData, setFormData] = useState({
    name: "",
    role: "",
    idNumber: "",
    aadharCard: [] as File[],
    roleIdCard: [] as File[],
    facePhoto1: [] as File[],
    facePhoto2: [] as File[],
    webcamPhoto: null as File | null, // Added for the optional webcam capture from the separate endpoint
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showWebcam, setShowWebcam] = useState(false);
  const webcamRef = useRef<Webcam>(null);

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  // This handles files from FileUpload components
  const handleFileChange = (field: string, files: File[]) => {
    // For single file fields like aadharCard, roleIdCard, facePhoto1, facePhoto2:
    // Ensure only the first file is taken if multiple are dropped/selected
    if (['aadharCard', 'roleIdCard', 'facePhoto1', 'facePhoto2'].includes(field)) {
      setFormData(prev => ({ ...prev, [field]: files.slice(0, 1) }));
    } else {
      setFormData(prev => ({ ...prev, [field]: files }));
    }
  };

  // This handles webcam photo capture specifically
  const capturePhoto = useCallback(async (photoField: 'facePhoto1' | 'facePhoto2') => {
    const imageSrc = webcamRef.current?.getScreenshot();
    if (imageSrc) {
      try {
        // Convert base64 to Blob, then to File object
        const response = await fetch(imageSrc);
        const blob = await response.blob();
        const file = new File([blob], `${formData.idNumber}_${photoField}.jpeg`, { type: 'image/jpeg' });

        // Update the form data with the captured file
        setFormData(prev => ({ ...prev, [photoField]: [file] })); // Ensure it's an array for FileUpload component consistency

        toast({
          title: "Photo Captured",
          description: `Photo for ${photoField.replace('facePhoto', 'Face Photo ')} captured successfully!`,
        });

        // If you intended to use the /webcam-capture for the *main* registration photo,
        // you would base64 encode it and send it here, but it's generally better
        // to send all registration data including images in one go via the /register endpoint.
        // If your backend specifically expects a separate endpoint for webcam photos
        // and you want to trigger that here:
        /*
        const base64Image = imageSrc.split(',')[1];
        const webcamCaptureResponse = await fetch(`${REGISTRATION_API_PROXY_URL}/webcam-capture`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: formData.idNumber, image_data: base64Image }),
        });
        if (!webcamCaptureResponse.ok) {
            const errorData = await webcamCaptureResponse.json();
            throw new Error(errorData.detail || "Failed to upload webcam photo");
        }
        */

      } catch (error) {
        console.error("Error capturing or processing webcam photo:", error);
        toast({
          title: "Photo Capture Failed",
          description: error instanceof Error ? error.message : "An error occurred during photo capture.",
          variant: "destructive",
        });
      }
    }
  }, [formData.idNumber, toast]);


  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);

    try {
      // Validate form fields
      if (!formData.name || !formData.role || !formData.idNumber) {
        throw new Error("Please fill in all required text fields.");
      }
      if (formData.aadharCard.length === 0) {
        throw new Error("Please upload Aadhar Card.");
      }
      if (formData.roleIdCard.length === 0) {
        throw new Error("Please upload Role ID Card.");
      }
      if (formData.facePhoto1.length === 0) {
        throw new Error("Please provide Face Photo 1.");
      }
      if (formData.facePhoto2.length === 0) {
        throw new Error("Please provide Face Photo 2.");
      }

      const formDataToSend = new FormData();
      formDataToSend.append("name", formData.name);
      formDataToSend.append("role", formData.role);
      formDataToSend.append("id_number", formData.idNumber);

      // Append files to FormData
      // FastAPI expects single files for aadhar_card, role_id_card, face_photo_1, face_photo_2
      formDataToSend.append("aadhar_card", formData.aadharCard[0]);
      formDataToSend.append("role_id_card", formData.roleIdCard[0]);
      formDataToSend.append("face_photo_1", formData.facePhoto1[0]);
      formDataToSend.append("face_photo_2", formData.facePhoto2[0]);

      // If you decide to include a separate webcamPhoto field for the main registration
      // and want to send it as an optional third face photo:
      /*
      if (formData.webcamPhoto) {
          formDataToSend.append("webcam_photo", formData.webcamPhoto);
      }
      */

      console.log("Attempting to send registration data via Node.js proxy to FastAPI:", {
        name: formData.name,
        role: formData.role,
        id_number: formData.idNumber,
        aadharCardFile: formData.aadharCard[0]?.name,
        roleIdCardFile: formData.roleIdCard[0]?.name,
        facePhoto1File: formData.facePhoto1[0]?.name,
        facePhoto2File: formData.facePhoto2[0]?.name,
        // webcamPhotoFile: formData.webcamPhoto?.name, // uncomment if using the optional webcamPhoto field
      });

      // CHANGED: Call Node.js backend proxy endpoint instead of direct FastAPI
      const response = await fetch(`${REGISTRATION_API_PROXY_URL}/register`, {
        method: 'POST',
        // When sending FormData, the 'Content-Type' header is usually
        // automatically set to 'multipart/form-data' by the browser.
        // Do NOT manually set 'Content-Type' here, as it will break the boundary string.
        body: formDataToSend,
      });

      const result = await response.json(); // Parse the JSON response from FastAPI (via Node.js)
      console.log("Node.js Proxy Response (from FastAPI):", result);

      if (!response.ok) {
        // If the response status is not 2xx, it's an error
        // Errors from FastAPI (proxied through Node.js) might have 'detail' or 'message'
        const errorMessage = result.detail || result.message || "An unknown error occurred during registration.";
        throw new Error(errorMessage);
      }

      toast({
        title: "Registration Successful",
        description: result.message || `User '${formData.name}' registered successfully!`,
      });

      // Reset form on success
      setFormData({
        name: "",
        role: "",
        idNumber: "",
        aadharCard: [],
        roleIdCard: [],
        facePhoto1: [],
        facePhoto2: [],
        webcamPhoto: null,
      });
      setShowWebcam(false); // Hide webcam after successful registration

    } catch (error) {
      console.error("Registration submission error:", error);
      toast({
        title: "Registration Failed",
        description: error instanceof Error ? error.message : "An unexpected error occurred. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  // Check if all required fields (text and files) are filled
  const isFormValid =
    formData.name.trim() !== "" &&
    formData.role.trim() !== "" &&
    formData.idNumber.trim() !== "" &&
    formData.aadharCard.length > 0 &&
    formData.roleIdCard.length > 0 &&
    formData.facePhoto1.length > 0 &&
    formData.facePhoto2.length > 0;

  return (
    <div className="min-h-full bg-[#121821]">
      <TopButtons /> {/* Add the TopButtons component here */}
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold bg-white bg-clip-text text-transparent mb-2">
              Personnel Registration
            </h1>
            <p className="text-muted-foreground">
              Register new personnel for campus access control
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-8">
            {/* Basic Information */}
            <Card className="bg-[#1F2733] border-[#1A222D] p-6">
              <h2 className="text-xl font-semibold mb-6 flex items-center space-x-2">
                <User className="h-5 w-5 text-white" />
                <span>Basic Information</span>
              </h2>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <Label htmlFor="name">Full Name *</Label>
                  <Input
                    id="name"
                    value={formData.name}
                    onChange={(e) => handleInputChange("name", e.target.value)}
                    placeholder="Enter full name"
                    className="bg-[#171D26] border-[#1F2733]"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="idNumber">ID Number *</Label>
                  <Input
                    id="idNumber"
                    value={formData.idNumber}
                    onChange={(e) => handleInputChange("idNumber", e.target.value)}
                    placeholder="Enter ID number"
                    className="bg-[#171D26] border-[#1F2733]"
                  />
                </div>

                <div className="space-y-2 md:col-span-2">
                  <Label htmlFor="role">Role *</Label>
                  <Select value={formData.role} onValueChange={(value) => handleInputChange("role", value)}>
                    <SelectTrigger className="bg-[#171D26] border-[#1F2733]">
                      <SelectValue placeholder="Select role" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="student">Student</SelectItem>
                      <SelectItem value="professor">Professor</SelectItem>
                      <SelectItem value="guard">Security Guard</SelectItem>
                      <SelectItem value="maintenance">Maintenance Staff</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </Card>

            {/* Document Uploads */}
            <Card className="bg-[#1F2733] border-[#1A222D] p-6">
              <h2 className="text-xl font-semibold mb-6 flex items-center space-x-2">
                <Upload className="h-5 w-5 text-white" />
                <span>Document Uploads</span>
              </h2>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <FileUpload
                  accept="image/*,.pdf"
                  label="Aadhar Card *"
                  onFileSelect={(files) => handleFileChange("aadharCard", files)}
                  maxSize={5}
                  files={formData.aadharCard} // Pass current files to FileUpload for display
                />

                <FileUpload
                  accept="image/*,.pdf"
                  label="Role ID Card *"
                  onFileSelect={(files) => handleFileChange("roleIdCard", files)}
                  maxSize={5}
                  files={formData.roleIdCard} // Pass current files to FileUpload for display
                />
              </div>
            </Card>

            {/* Face Photos */}
            <Card className="bg-[#1F2733] border-[#1A222D] p-6">
              <h2 className="text-xl font-semibold mb-6 flex items-center space-x-2">
                <Camera className="h-5 w-5 text-white" />
                <span>Face Recognition Photos</span>
              </h2>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                <FileUpload
                  accept="image/*"
                  label="Face Photo 1 *"
                  onFileSelect={(files) => handleFileChange("facePhoto1", files)}
                  maxSize={5}
                  files={formData.facePhoto1} // Pass current files to FileUpload for display
                />

                <FileUpload
                  accept="image/*"
                  label="Face Photo 2 *"
                  onFileSelect={(files) => handleFileChange("facePhoto2", files)}
                  maxSize={5}
                  files={formData.facePhoto2} // Pass current files to FileUpload for display
                />
              </div>

              {/* Webcam Capture Section */}
              <div className="border-t border-border pt-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-medium">Live Capture</h3>
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => setShowWebcam(!showWebcam)}
                  >
                    <Camera className="h-4 w-4 mr-2 text-white" />
                    {showWebcam ? "Hide Camera" : "Use Camera"}
                  </Button>
                </div>

                {showWebcam && (
                  <Card className="bg-card/50 border-white/20 p-4">
                    <div className="aspect-video bg-muted rounded-lg overflow-hidden">
                      <Webcam
                        ref={webcamRef}
                        audio={false}
                        screenshotFormat="image/jpeg"
                        videoConstraints={{
                          width: 1280,
                          height: 720,
                          facingMode: "user"
                        }}
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <div className="flex justify-center mt-4 space-x-4">
                      <Button
                        type="button"
                        variant="outline"
                        onClick={() => capturePhoto('facePhoto1')}
                        disabled={!webcamRef.current}
                      >
                        <Camera className="h-4 w-4 mr-2" />
                        Capture Photo 1
                      </Button>
                      <Button
                        type="button"
                        variant="outline"
                        onClick={() => capturePhoto('facePhoto2')}
                        disabled={!webcamRef.current}
                      >
                        <Camera className="h-4 w-4 mr-2" />
                        Capture Photo 2
                      </Button>
                    </div>
                  </Card>
                )}
              </div>
            </Card>

            {/* Form Status */}
            <Card className="bg-[#1F2733] border-[#1A222D] p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <CheckCircle className={`h-6 w-6 ${isFormValid ? 'text-white' : 'text-muted-foreground'}`} />
                  <div>
                    <p className="font-medium">Registration Form</p>
                    <p className="text-sm text-muted-foreground">
                      {isFormValid ? "Ready to submit" : "Please complete all required fields (marked with *)"}
                    </p>
                  </div>
                </div>

                <Badge variant={isFormValid ? "default" : "secondary"}>
                  {isFormValid ? "Valid" : "Incomplete"}
                </Badge>
              </div>
            </Card>

            {/* Submit Button */}
            <div className="flex justify-center">
              <Button
                type="submit"
                variant="cyber"
                size="xl"
                disabled={!isFormValid || isSubmitting}
                className="min-w-[200px]"
              >
                {isSubmitting ? "Registering..." : "Register Personnel"}
              </Button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}