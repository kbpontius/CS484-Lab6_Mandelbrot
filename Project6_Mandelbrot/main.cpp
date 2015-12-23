/*
 *
 * File:            mandelbrot.cpp
 * Author:          Dany Shaanan
 * Website:         http://danyshaanan.com
 * File location:   https://github.com/danyshaanan/mandelbrot/blob/master/cpp/mandelbrot.cpp
 *
 * Created somewhen between 1999 and 2002
 * Rewritten August 2013
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "mpi.h"

using namespace std;

////////////////////////////////////////////////////////////////////////////////

/* MANDELBROT CONSTS */
const int MAX_WIDTH_HEIGHT = 30000;
const int HUE_PER_ITERATION = 5;
const bool DRAW_ON_KEY = true;
const int WIDTH_HEIGHT = 4000;
const int ZOOM = 1000;

////////////////////////////////////////////////////////////////////////////////

/* MPI CONSTS */
const int CHUNK_SIZE = 1000;
const int CHUNK_NUMBER_TOTAL = WIDTH_HEIGHT / CHUNK_SIZE;
const int STATUS_CHECK_TAG = 100;

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
        //centerX = -.75;
        //centerY = 0;
        centerX = -1.186340599860225;
        centerY = -0.303652988644423;
        zoom = ZOOM;
        maxIterations = 300;
        w = WIDTH_HEIGHT;
        h = WIDTH_HEIGHT;
    }
};

////////////////////////////////////////////////////////////////////////////////
void sendWork(int chunkNumber, double xs[MAX_WIDTH_HEIGHT], double ys[MAX_WIDTH_HEIGHT], unsigned char *img, int destination) {
    MPI_Request request;
//    MPI_Isend(xs, 1, MPI_DOUBLE, destination, 0, MPI_COMM_WORLD, &request);
//    MPI_Isend(ys, 1, MPI_DOUBLE, destination, 1, MPI_COMM_WORLD, &request);
//    MPI_Isend(img, 1, MPI_CHAR, destination, 2, MPI_COMM_WORLD, &request);
    MPI_Isend(&chunkNumber, 1, MPI_INT, destination, 3, MPI_COMM_WORLD, &request);
}

int getStart(int tasksRemaining, int chunkSize) {
    return -1;
}

int getEnd(int tasksRemaining, int chunkSize) {
    return -1;
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

unsigned char *createImage(State state, int argc, char *argv[]) {
    int iproc, nproc;
    int w = state.w;
    int h = state.h;
    int chunksSent = 0;
    int responseChunks = 0;
    
    if (w > MAX_WIDTH_HEIGHT) w = MAX_WIDTH_HEIGHT;
    if (h > MAX_WIDTH_HEIGHT) h = MAX_WIDTH_HEIGHT;
    
    unsigned char r, g, b;
    unsigned char *img = NULL;
    
    if (img) free(img);
    
    long long size = (long long)w*(long long)h*3;
//    printf("Malloc w %zu, h %zu, %zu\n", w, h, size);
    img = (unsigned char *)malloc(size);
//    printf("malloc returned %X\n", img);
    
    double xs[MAX_WIDTH_HEIGHT], ys[MAX_WIDTH_HEIGHT];
    
    MPI_Status status;
    MPI_Init(&argc, &argv);
    
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &iproc);
    
    if (iproc == 0) {
        int i;
        
        for (i = 1; i < nproc; i++) {
            if (chunksSent < CHUNK_NUMBER_TOTAL) {
                sendWork(chunksSent, xs, ys, img, i);
                fprintf(stderr, "%i: WORK(%i) SENT TO: %i\n", iproc, chunksSent,i);
                chunksSent++;
            }
        }
    }
    
    while (responseChunks < CHUNK_NUMBER_TOTAL) {
        if (iproc == 0) {
            fprintf(stderr, "%i: AWAITING RESPONSE FROM ANYSOURCE\n", iproc);
            double newXS[MAX_WIDTH_HEIGHT], newYS[MAX_WIDTH_HEIGHT];
            unsigned char *newImg = NULL;
            if (newImg) free(newImg);
            int newChunkNumber;
//            MPI_Recv(newXS, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
//            MPI_Recv(newYS, 1, MPI_DOUBLE, status.MPI_SOURCE, 1, MPI_COMM_WORLD, &status);
//            MPI_Recv(newImg, 1, MPI_CHAR, status.MPI_SOURCE, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&newChunkNumber, 1, MPI_INT, MPI_ANY_SOURCE, 3, MPI_COMM_WORLD, &status);
            MPI_Send(&chunksSent, 1, MPI_INT, status.MPI_SOURCE, STATUS_CHECK_TAG, MPI_COMM_WORLD);
            responseChunks++;
            
            fprintf(stderr, "%i: RESPONSE FROM: %i (%i/%i)\n", iproc, status.MPI_SOURCE, newChunkNumber + 1, chunksSent);
            
            // TODO: INCORPORATE UPDATES!!
            
            if (chunksSent < CHUNK_NUMBER_TOTAL) {
                sendWork(chunksSent, xs, ys, img, status.MPI_SOURCE);
                fprintf(stderr, "%i: WORK(%i) SENT TO: %i\n", iproc, chunksSent, status.MPI_SOURCE);
                chunksSent++;
                
                fprintf(stderr, "%i: CURRENT STATUS --> RESPONSE_CHUNKS: %i, CHUNKS_SENT: %i\n", iproc, responseChunks, chunksSent);
            }
        } else {
            fprintf(stderr, "%i: AWAITING WORK FROM 0\n", iproc);
//            MPI_Recv(xs, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
//            MPI_Recv(ys, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
//            MPI_Recv(img, 1, MPI_CHAR, 0, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&chunksSent, 1, MPI_INT, 0, 3, MPI_COMM_WORLD, &status);
            fprintf(stderr, "%i: RECEIVED WORK, CHUNK #: %i\n", iproc, chunksSent);
            
            int start = chunksSent * CHUNK_SIZE;
            int end = start + CHUNK_SIZE;
            int px;
            
            for (px = start; px < end; px++) {
                xs[px] = (px - w/2)/state.zoom + state.centerX;
            }
            
            int py;
            for (py = start; py < end; py++) {
                ys[py] = (py - h/2)/state.zoom + state.centerY;
            }
            
            for (px = start; px < end; px++) {
                for (int py = start; py < end; py++) {
                    r = g = b = 0;
                    float iterations = iterationsToEscape(xs[px], ys[py], state.maxIterations);
                    
                    if (iterations != -1) {
                        float h = HUE_PER_ITERATION * iterations;
                        r = hue2rgb(h + 120);
                        g = hue2rgb(h);
                        b = hue2rgb(h + 240);
                    }
                    
                    long long loc = ((long long)px+(long long)py*(long long)w)*3;
                    img[loc + 2] = (unsigned char)(r);
                    img[loc + 1] = (unsigned char)(g);
                    img[loc + 0] = (unsigned char)(b);
                }
            }
            
            fprintf(stderr, "%i: FINSIHED WORK FOR CHUNK #: %i\n", iproc, chunksSent);
//            MPI_Send(xs, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
//            MPI_Send(ys, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
//            MPI_Send(img, 1, MPI_CHAR, 0, 2, MPI_COMM_WORLD);
            MPI_Send(&chunksSent, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
            MPI_Recv(&responseChunks, 1, MPI_INT, 0, STATUS_CHECK_TAG, MPI_COMM_WORLD, &status);
            
            fprintf(stderr, "%i: ---> RESPONSE CHUNK RECEIVED: %i, %i\n", iproc, responseChunks, CHUNK_NUMBER_TOTAL);
        }
    }
    
    fprintf(stderr, "%i: EXECUTION COMPLETE, EXITING!\n", iproc);
    
    MPI_Finalize();
    
    return img;
}

////////////////////////////////////////////////////////////////////////////////

void draw(State state, int argc, char *argv[]) {
    unsigned char *img = createImage(state, argc, argv);
    writeImage(img, state.w, state.h);
}

double When()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + (double) tp.tv_usec * 1e-6);
}

int main(int argc, char *argv[]) {
    State state;
    double startTime = When();
    draw(state, argc, argv);
    double endTime = When();
    
    double totalTime = endTime - startTime;
    fprintf(stderr, "TOTAL TIME: %f\n", totalTime);
    
    return 0;
}
