// src/components/upload/FileUpload.tsx

import { useRef, useState, useEffect } from "react"; // Import useEffect
import { Upload, X, File, Image } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface FileUploadProps {
  accept?: string;
  multiple?: boolean;
  onFileSelect: (files: File[]) => void;
  label: string;
  className?: string;
  maxSize?: number; // in MB
  files?: File[]; // Add this new prop to receive initial/controlled files
}

export const FileUpload = ({
  accept = "*/*",
  multiple = false,
  onFileSelect,
  label,
  className,
  maxSize = 10,
  files: controlledFiles = [], // New prop, default to empty array
}: FileUploadProps) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [internalFiles, setInternalFiles] = useState<File[]>(controlledFiles); // Use internalFiles for component's state
  const [error, setError] = useState<string>("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Use useEffect to synchronize internalFiles with controlledFiles prop
  useEffect(() => {
    // Only update if the controlledFiles array has actually changed content
    // This is a shallow comparison, for deep comparison you might need a more complex memoization
    if (controlledFiles !== internalFiles) { // Basic check for reference equality
        setInternalFiles(controlledFiles);
    }
  }, [controlledFiles]);


  const validateFile = (file: File): string | null => {
    if (file.size > maxSize * 1024 * 1024) {
      return `File size must be less than ${maxSize}MB`;
    }
    return null;
  };

  const handleFiles = (newFiles: FileList | null) => {
    if (!newFiles) return;

    const filesArray = Array.from(newFiles);
    let validFiles: File[] = [];
    let errorMsg = "";

    for (const file of filesArray) {
      const error = validateFile(file);
      if (error) {
        errorMsg = error;
        break; // Stop on first error for single file selection
      }
      validFiles.push(file);
    }

    if (errorMsg) {
      setError(errorMsg);
      // Also clear any previously selected files if an error occurs and it's a single-file upload
      if (!multiple) {
          setInternalFiles([]);
          onFileSelect([]);
      }
      return;
    }

    setError("");
    const finalFiles = multiple ? [...internalFiles, ...validFiles] : validFiles; // Use internalFiles here
    setInternalFiles(finalFiles);
    onFileSelect(finalFiles);
  };

  const removeFile = (index: number) => {
    const newFiles = internalFiles.filter((_, i) => i !== index); // Use internalFiles here
    setInternalFiles(newFiles);
    onFileSelect(newFiles);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    handleFiles(e.dataTransfer.files);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const getFileIcon = (file: File) => {
    if (file.type.startsWith("image/")) {
      // For images, show a preview if possible, or the image icon
      // You might add a small image preview here in a real app
      return <Image className="h-8 w-8 text-white" />;
    }
    return <File className="h-8 w-8 text-primary" />;
  };

  return (
    <div className={cn("space-y-4", className)}>
      <Card
        className={cn(
          "transition-all duration-300 cursor-pointer",
          isDragOver
            ? "border-[#121821] bg-primary/10"
            : "border-[#121821]",
          error && "border-destructive",
          internalFiles.length > 0 && "border-[#1B2530]" // Indicate when files are selected
        )}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={handleClick}
      >
        <div className="p-8 text-center bg-[#101921]">
          <div className="relative inline-block mb-4">
            <Upload className={cn(
              "h-12 w-12 mx-auto transition-colors",
              isDragOver ? "text-primary" : "text-muted-foreground"
            )} />
            {isDragOver && (
              <div className="absolute inset-0 bg-primary/20 rounded-full blur-lg" />
            )}
          </div>

          <h3 className="text-lg font-semibold mb-2">{label}</h3>
          <p className="text-muted-foreground mb-4">
            Drag and drop files here or click to browse
          </p>
          <p className="text-sm text-muted-foreground">
            Max file size: {maxSize}MB
          </p>
        </div>
      </Card>

      <input
        ref={fileInputRef}
        type="file"
        accept={accept}
        multiple={multiple}
        onChange={(e) => handleFiles(e.target.files)}
        className="hidden"
      />

      {error && (
        <div className="text-destructive text-sm font-medium">{error}</div>
      )}

      {internalFiles.length > 0 && ( // Use internalFiles here
        <div className="space-y-2">
          <h4 className="font-medium text-sm text-foreground">Selected Files:</h4>
          {internalFiles.map((file, index) => ( // Use internalFiles here
            <Card key={index} className="p-3 border-[#1B2530]">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  {getFileIcon(file)}
                  <div>
                    <p className="font-medium text-sm">{file.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    removeFile(index);
                  }}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};