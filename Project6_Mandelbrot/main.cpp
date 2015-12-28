#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>
#include "mpi.h"

using namespace std;

////////////////////////////////////////////////////////////////////////////////

/* MANDELBROT CONSTS */
const int MAX_WIDTH_HEIGHT = 30000;
const int HUE_PER_ITERATION = 5;
const int WIDTH_HEIGHT = 28000;
const int ZOOM = 1000;

////////////////////////////////////////////////////////////////////////////////

/* MPI CONSTS */
const int CHUNK_SIZE = 2000;
const int CHUNK_NUMBER_TOTAL = WIDTH_HEIGHT / CHUNK_SIZE;
const int TAG_STATUS_CHECK = 100;
const int TAG_EARLY_TERMINATION = 99;

////////////////////////////////////////////////////////////////////////////////

class State {
public:
    double centerX;
    double centerY;
    double zoom;
    int maxIterations;
    int w;
    int h;
    State() {
        centerX = -1.186340599860225;
        centerY = -0.303652988644423;
        zoom = ZOOM;
        maxIterations = 300;
        w = WIDTH_HEIGHT;
        h = WIDTH_HEIGHT;
    }
};

////////////////////////////////////////////////////////////////////////////////
double When()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + (double) tp.tv_usec * 1e-6);
}

float iterationsToEscape(double x, double y, int maxIterations) {
    double tempa;
    double a = 0;
    double b = 0;
    
    for (int i = 0 ; i < maxIterations ; i++) {
        tempa = (a * a) - (b * b) + x;
        b = (2 * a * b) + y;
        a = tempa;
        
        if ((a * a) + (b * b) > 64) {
            return i - log(sqrt(a*a+b*b))/log(8);
        }
    }
    
    return -1;
}

int hue2rgb(float t){
    while (t > 360) {
        t -= 360;
    }
    if (t < 60) return 255. * t / 60.;
    if (t < 180) return 255;
    if (t < 240) return 255. * (4. - t / 60.);
    return 0;
}

void writeImage(unsigned char *img, int w, int h) {
    long long filesize = 54 + 3*(long long)w*(long long)h;
    unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
    unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
    unsigned char bmppad[3] = {0,0,0};
    
    bmpfileheader[ 2] = (unsigned char)(filesize    );
    bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
    bmpfileheader[ 4] = (unsigned char)(filesize>>16);
    bmpfileheader[ 5] = (unsigned char)(filesize>>24);
    
    bmpinfoheader[ 4] = (unsigned char)(       w    );
    bmpinfoheader[ 5] = (unsigned char)(       w>> 8);
    bmpinfoheader[ 6] = (unsigned char)(       w>>16);
    bmpinfoheader[ 7] = (unsigned char)(       w>>24);
    bmpinfoheader[ 8] = (unsigned char)(       h    );
    bmpinfoheader[ 9] = (unsigned char)(       h>> 8);
    bmpinfoheader[10] = (unsigned char)(       h>>16);
    bmpinfoheader[11] = (unsigned char)(       h>>24);
    
    fprintf(stderr, "WRITING IMAGE TO FILE\n");
    
    FILE *f;
    f = fopen("temp.bmp","wb");
    fwrite(bmpfileheader,1,14,f);
    fwrite(bmpinfoheader,1,40,f);
    
    for (int i=0; i<h; i++) {
        long long offset = ((long long)w*(h-i-1)*3);
        fwrite(img+offset,3,w,f);
        fwrite(bmppad,1,(4-(w*3)%4)%4,f);
    }
    
    fclose(f);
}

// This method sends asynchronous work requests.
void sendWork(int chunkNumber, int destination) {
    MPI_Request request;
    int isTerminated = 0;
    
    MPI_Isend(&isTerminated, 1, MPI_INT, destination, TAG_EARLY_TERMINATION, MPI_COMM_WORLD, &request);
    MPI_Isend(&chunkNumber, 1, MPI_INT, destination, 3, MPI_COMM_WORLD, &request);
}

void sendTermination(int destination) {
    MPI_Request request;
    int isTerminated = 1;
    MPI_Isend(&isTerminated, 1, MPI_INT, destination, TAG_EARLY_TERMINATION, MPI_COMM_WORLD, &request);
}

double createImage(State state, int argc, char *argv[], double startTime) {
    int iproc, nproc;
    int w = state.w;
    int h = state.h;
    int chunksSent = 0;
    int responseChunks = 0;
    double endTime = 0.0;
    int isTerminated = 1;
    long double sizeOfImg;
    
    if (w > MAX_WIDTH_HEIGHT) w = MAX_WIDTH_HEIGHT;
    if (h > MAX_WIDTH_HEIGHT) h = MAX_WIDTH_HEIGHT;
    
    unsigned char r, g, b;
    unsigned char *img = NULL;
    
    if (img) free(img);
    
    long long size = (long long)w*(long long)h*3;
    img = (unsigned char *)malloc(size);
    sizeOfImg = (double)(size / CHUNK_NUMBER_TOTAL);
    
//    fprintf(stderr, "--->>>SIZE OF IMAGE: %f\n", sizeOfImg);
    
    double xs[MAX_WIDTH_HEIGHT], ys[MAX_WIDTH_HEIGHT];
    
    MPI_Status status;
    MPI_Init(&argc, &argv);
    
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &iproc);
    
//    fprintf(stderr, "HELLO FROM iproc: %i of %i\n", iproc, nproc);
    
    if (iproc == 0) {
        int i;
        
        /*
            Farm out initial workload to available nodes.
        */
        for (i = 1; i < nproc; i++) {
            if (chunksSent < CHUNK_NUMBER_TOTAL) {
                sendWork(chunksSent, i);
//                fprintf(stderr, "%i: WORK(%i) SENT TO: %i\n", iproc, chunksSent,i);
                chunksSent++;
            } else {
                sendTermination(i);
            }
        }
    }
    
    /*
        Repeat until all work has been 
        assigned and finished.
    */
    while (responseChunks < CHUNK_NUMBER_TOTAL) {
        if (iproc == 0) {
//            fprintf(stderr, "%i: AWAITING RESPONSE FROM ANYSOURCE\n", iproc);
            
            // Setup temporary variables for receiving data.
            unsigned char *newImg = NULL;
            if (newImg) free(newImg);
            newImg = (unsigned char *)malloc(size);
            int nodeStartNumber;
            int nodeEndNumber;
            
            // Wait for response from node, then respond with work count remaining (chunksSent).
            MPI_Recv(newImg, sizeOfImg, MPI_UNSIGNED_CHAR, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&nodeStartNumber, 1, MPI_INT, status.MPI_SOURCE, 3, MPI_COMM_WORLD, &status);
            MPI_Send(&chunksSent, 1, MPI_INT, status.MPI_SOURCE, TAG_STATUS_CHECK, MPI_COMM_WORLD);
            
            // Received another finished chunk of the image.
            responseChunks++;
            
//            fprintf(stderr, "%i: RESPONSE FROM: %i (%i/%i)\n", iproc, status.MPI_SOURCE, nodeStartNumber + 1, chunksSent);
            
            /*
                Update the master image.
            */
            nodeStartNumber = nodeStartNumber * CHUNK_SIZE;
            nodeEndNumber = nodeStartNumber + CHUNK_SIZE;
            
            fprintf(stderr, "%i: >>>>>>>> START: %i || END: %i <<<<<<<<\n", iproc, nodeStartNumber, nodeEndNumber);
            
            int i, j;
            for (i = nodeStartNumber; i <= nodeEndNumber; i++) {
                for (j = 0; j < WIDTH_HEIGHT; j++) {
                    long long loc = ((long long)i+(long long)j*(long long)w)*3;
                    long long newLoc = ((long long)(i - nodeStartNumber)+(long long)j*(long long)w)*3;
                    img[loc + 2] = newImg[newLoc + 2];
                    img[loc + 1] = newImg[newLoc + 1];
                    img[loc + 0] = newImg[newLoc + 0];
                }
            }
            
            /*
                Check whether more data chunks are left to work on.
                This is where the the master-slave paradigm has been implemented. The
                root process determines whether there is more work, then continues to
                push out work as long as its available.
            */
            if (chunksSent < CHUNK_NUMBER_TOTAL) {
                sendWork(chunksSent, status.MPI_SOURCE);
//                fprintf(stderr, "%i: WORK(%i) SENT TO: %i\n", iproc, chunksSent, status.MPI_SOURCE);
                chunksSent++;
//                fprintf(stderr, "%i: CURRENT STATUS --> RESPONSE_CHUNKS: %i, CHUNKS_SENT: %i\n", iproc, responseChunks, chunksSent);
            }
        } else {
            // Setup temporary variables for receiving data.
            unsigned char *newImg = NULL;
            if (newImg) free(newImg);
            newImg = (unsigned char *)malloc(sizeOfImg);
            
//            fprintf(stderr, "%i: AWAITING WORK FROM 0\n", iproc);
            
            /*
                -- CHECK FOR EARLY TERMINATION
                This condition comes up when there are more nodes than work to be 
                done in the first round of work assignments.
            */
            if (isTerminated == 1) {
                MPI_Recv(&isTerminated, 1, MPI_INT, 0, TAG_EARLY_TERMINATION, MPI_COMM_WORLD, &status);
                
                if (isTerminated == 1) {
//                    fprintf(stderr, ">>>> %i: TERMINATING EARLY\n", iproc);
                    break;
                }
            }
            
            MPI_Recv(&chunksSent, 1, MPI_INT, 0, 3, MPI_COMM_WORLD, &status);
            fprintf(stderr, "%i: RECEIVED WORK, CHUNK #: %i\n", iproc, chunksSent);
            
            int start = CHUNK_SIZE * chunksSent;
            int end = start + CHUNK_SIZE;
            int px;
            
            for (px = start; px <= end; px++) {
                xs[px] = (px - w/2)/state.zoom + state.centerX;
            }
            
            int py;
            for (py = 0; py < WIDTH_HEIGHT; py++) {
                ys[py] = (py - h/2)/state.zoom + state.centerY;
            }
            
            for (px = start; px <= end; px++) {
                for (int py = 0; py < WIDTH_HEIGHT; py++) {
                    r = g = b = 0;
                    float iterations = iterationsToEscape(xs[px], ys[py], state.maxIterations);
                    
                    if (iterations != -1) {
                        float h = HUE_PER_ITERATION * iterations;
                        r = hue2rgb(h + 120);
                        g = hue2rgb(h);
                        b = hue2rgb(h + 240);
                    }
                    
                    long long loc = ((long long)(px - start)+(long long)py*(long long)w)*3;
                    newImg[loc + 2] = (unsigned char)(r);
                    newImg[loc + 1] = (unsigned char)(g);
                    newImg[loc + 0] = (unsigned char)(b);
                }
            }
            
//            fprintf(stderr, "%i: FINISHED WORK FOR CHUNK #: %i\n", iproc, chunksSent);
            
            MPI_Send(newImg, sizeOfImg, MPI_UNSIGNED_CHAR, 0, 2, MPI_COMM_WORLD);
            MPI_Send(&chunksSent, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
            MPI_Recv(&responseChunks, 1, MPI_INT, 0, TAG_STATUS_CHECK, MPI_COMM_WORLD, &status);
            
//            fprintf(stderr, "%i: ---> RESPONSE CHUNK RECEIVED: %i, %i\n", iproc, responseChunks, CHUNK_NUMBER_TOTAL);
        }
    }
    
    // Get the final endTime.
    if (iproc == 0) {
        endTime = When();
//        fprintf(stderr, "%i: -------- FINISHED IMAGE CALCULATION, WRITING TO DISK (endTime: %f) --------\n", iproc, endTime);
        writeImage(img, WIDTH_HEIGHT, WIDTH_HEIGHT);
    }
    
//    fprintf(stderr, "%i: ------------------------------ EXITING PROGRAM ------------------------------\n", iproc);
    
    MPI_Finalize();
    
    fprintf(stderr, "%i: ------------------------------ RETURNING! ------------------------------\n", iproc);
    
    if (iproc == 0) {
        double executionTime = endTime - startTime;
        return executionTime;
    }
    
    return -1;
}

////////////////////////////////////////////////////////////////////////////////
double draw(State state, int argc, char *argv[]) {
    double startTime = When();
    return createImage(state, argc, argv, startTime);
}

int main(int argc, char *argv[]) {
    State state;
    double totalTime = draw(state, argc, argv);
    
    if (totalTime > 0) {
        fprintf(stderr, "TOTAL TIME: %f\n", totalTime);
    }
    
    return 0;
}
