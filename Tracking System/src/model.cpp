#include "../include/model.h"
#define ROWS 720

#define COLS 1280
using namespace std;
    //threshold for differenece between 2 angles,2 slopes ....etc
    const FLOAT threshold = 0.087f;

    const FLOAT zero_threshold = 0.001; 
    //threshold for small forces
    //used in insertion







    void swap(point& p1, point& p2) {
        point temp = p1;
        p1 = p2;
        p2 = temp;
    }

    /*
    point class
    */


    FLOAT point::normal_theta(const point  src_pt)const {
        FLOAT dy = src_pt.y - y;
        FLOAT dx = src_pt.x - x;
        return atan2(dy, dx) +M_PI/2;
    }
    FLOAT point::theta(const point src_pt)const {
        FLOAT dy = src_pt.y - y;
        FLOAT dx = src_pt.x - x;
        return atan2(dy, dx);
    }

    
    /*
    model class
    */

//allocates constant size for the model (30)
model::model(void){
    vec = new point[max_model_size];    
    temp_vec = new FLOAT[max_model_size] ;  

    force_vec= new FLOAT[max_forces_size] ; 
    elements_size = 0 ; 
}

//destructor 
model::~model(void){
    delete[]vec ;
    vec= NULL ;
    delete[]force_vec ; 
    force_vec = NULL; 
    delete[]temp_vec ;
    temp_vec =NULL; 
    elements_size= 0;
}
//returns number of elements in the model
UINT32 model::get_size(void)const{
    return elements_size; 
}

//inserts a point at an index
BOOL model::insert(const point&new_pt, UINT32 index){
    //ignore if more than maximum model size 
    if(index<=elements_size&&elements_size<max_model_size){
        //shift elements to the right
        for(UINT32 i =elements_size ;i>index;i--){
            vec[i] = vec[i-1];
        }
        //insert new point
        vec[index] = new_pt ;
        elements_size++; 
        return 1 ;
    }
    return 0;
}

BOOL model::remove(UINT32 index){
    if(index<elements_size&&elements_size>0){
        for(UINT32 i =index; i<elements_size;i++){
            vec[i]=vec[i+1] ;
        }
        elements_size--; 
        return 1; 
    }
    return 0 ; 
}

//constant accessor
point& model:: operator[](UINT32 index)const{
    if(index<elements_size){
        return vec[index] ;
    }
}
//variable accessor
point& model:: operator[](UINT32 index){
    if(index<elements_size){
        return vec[index] ;
    }
}
//reinit
void model::reinit(const point&pt){
    elements_size = 4 ; 
    vec[0] = point((pt.x-20),(pt.y-20));
    vec[1] = point((pt.x+20),(pt.y-20));
    vec[2] = point((pt.x+20),(pt.y+20));
    vec[3] = point((pt.x-20),(pt.y+20));    
    for(UINT32 i=0;i<4 ;i++){
        if(vec[i].x<0){
            vec[i].x= 0 ; 
        }
        if(vec[i].x>COLS){
            vec[i].x= COLS -1;
        }
        if(vec[i].y<0){
            vec[i].y= 0 ; 
        }
        if(vec[i].y>ROWS){
            vec[i].y= ROWS -1;
        }
    }
}   

FLOAT model::area(void)const{
    FLOAT ret_area = 0  ;
    for(UINT32 i=0; i<elements_size;i++){
        ret_area +=(vec[i].x*vec[(i+1)%elements_size].y-vec[(i+1)%elements_size].x*vec[i].y);
    }
    return fabs(ret_area/2);
}

point model::centroid(void)const {
    point ret_point(0,0);
    for(UINT32 i=0 ;i<elements_size;i++){
        INT32 val =(vec[i].x*vec[(i+1)%elements_size].y-vec[(i+1)%elements_size].x*vec[i].y) ;
        ret_point.x+=(vec[i].x+vec[(i+1)%elements_size].x)*val;
        ret_point.y+=(vec[i].y+vec[(i+1)%elements_size].y)*val;
    }
    FLOAT poly_area = area() ;
    ret_point.x = round(fabs(static_cast<FLOAT>(ret_point.x)/(FLOAT(6)*poly_area))) ;
    ret_point.y = round(fabs(static_cast<FLOAT>(ret_point.y)/(FLOAT(6)*poly_area))) ;
    return ret_point ;
}
point model::centroid_average(void)const{
    if(elements_size>0){
        point ret_point(0,0);
        for(UINT32 i= 0  ;i<elements_size;i++){
            ret_point.x+=vec[i].x ;
            ret_point.y+=vec[i].y ;
        }
        ret_point.x= round(static_cast<FLOAT>(ret_point.x)/static_cast<FLOAT>(elements_size));
        ret_point.y= round(static_cast<FLOAT>(ret_point.y)/static_cast<FLOAT>(elements_size));
        return ret_point ;
    }
    return point(0,0) ;
}
    
BOOL model::forces(INT32* hist){
    if(hist){
        for(UINT32 i =0 ; i <elements_size;i++){
            FLOAT lambda = vec[i].length(vec[(i+1)%elements_size])*2 ;
            force_vec[2 * i] = static_cast<FLOAT>(hist[8 * i + A + 1] - hist[8 * i + D + 0]) / lambda;
            if (((float)hist[8 * i + A + 1] / (float)(hist[8 * i + A + 1] + hist[8 * i + A + 0]) > 0.85)
                || ((float)hist[8 * i + D + 0] / (float)(hist[8 * i + D + 0] + hist[8 * i + D + 1]) > 0.85)) force_vec[2 * i] *= 6;

            force_vec[2 * i + 1] = static_cast<FLOAT>(hist[8 * i + B + 1] - hist[8 * i + C + 0]) / lambda;
            if (((float)hist[8 * i + B + 1] / (float)(hist[8 * i + B + 1] + hist[8 * i + B + 0]) > 0.85)
                || ((float)hist[8 * i + C + 0] / (float)(hist[8 * i + C + 0] + hist[8 * i + C + 1]) > 0.85)) force_vec[2 * i + 1] *= 6;
        }
        return 1;
    }
    return 0;
}


BOOL model::deform(INT32* hist){
    if(forces(hist)){
        for(UINT32 i=0 ; i <elements_size;i++){
            temp_vec[i]=vec[i].normal_theta(vec[(i+1)%elements_size]);
        }
        for(UINT32 i=0; i<elements_size;i++){
            INT32 prev_force_index = 2*i-1  ;
            INT32 prev_angle_index = i - 1; 
            if(prev_force_index <0){
                prev_force_index +=elements_size*2;
            }           
            if (prev_angle_index < 0) {
                prev_angle_index += elements_size ;
            }       
            vec[i].x -=round(force_vec[2*i]*cos(temp_vec[i])+
                             force_vec[prev_force_index]*cos(temp_vec[prev_angle_index]));
            vec[i].y -=round(force_vec[2*i]*sin(temp_vec[i])+
                             force_vec[prev_force_index]*sin( temp_vec[prev_angle_index]));
            if(vec[i].x<0){
                vec[i].x= 0 ; 
            }
            if(vec[i].x>=COLS){
                vec[i].x= COLS -1;
            }
            if(vec[i].y<0){
                vec[i].y= 0 ; 
            }
            if(vec[i].y>=ROWS){
                vec[i].y= ROWS -1;
            }
        }
        return 1 ; 
    }
    return 0 ;
}
void model::insert_vertecies_at_position(INT32* hist){
    for(UINT32 i =0 ; i<elements_size;i++){
        //don't exceed max size
        if(elements_size<max_model_size){
            //if forces on that edge are both minimum 
            if((fabs(force_vec[i])<=force_threshold)&&
               (fabs(force_vec[i+1])<=force_threshold)){
                //if a region intersecion with error is more than limit insert and skip added vertex
                if ((hist[8 * i + A + 1] > (limit * vec[i].length(vec[(i + 1) % elements_size]) / 2)) || (hist[8 * i + B + 1] > (limit * vec[i].length(vec[(i + 1) % elements_size]) / 2)) ||
                    (hist[8 * i + C + 0] > (limit * vec[i].length(vec[(i + 1) % elements_size]) / 2)) || (hist[8 * i + D + 0] > (limit * vec[i].length(vec[(i + 1) % elements_size]) / 2))) {
                    insert(vec[i].mid_point(vec[(i+1)%elements_size]),(i+1));
                    i++;
                }
            }
        }
    }
}

    BOOL model::remove_spikes(void) {
        for (UINT32 i = 0; i < elements_size; i++) {
            INT32 prev_index = (i - 1);
            if (prev_index < 0) {
                prev_index += elements_size;
            }
            FLOAT prev_theta = vec[prev_index].normal_theta(vec[i]);
            FLOAT current_theta = vec[i].normal_theta(vec[(i + 1) % elements_size]);
            //first make sure they aren't straight lines since the noisy point forms straight line with normal vertex
            if (fabs(M_PI - fabs(prev_theta - current_theta)) <= threshold) {
                remove(i);
                i--;
            }
        }
        return  1;
    }
    BOOL model::remove_small_lengths(void) {
        for (UINT32 i = 0; i < elements_size; i++) {
            if (vec[i].length(vec[(i + 1) % elements_size]) <length_threshold) {
                remove(i);
                i--;

            }
        }
        return 1;
    }

    BOOL model::remove_extra_vertices(void) {
        if (elements_size > 2) {
            remove_spikes();
            remove_verticies_in_middle();
            remove_small_lengths();
            return TRUE;
        }
        return FALSE;
    }
    BOOL model::remove_verticies_in_middle(void) {
        for (UINT32 i = 0; i < elements_size; i++) {
            INT32 prev_index = (i - 1);
            if (prev_index < 0) {
                prev_index += elements_size;
            }
            FLOAT prev_theta = vec[prev_index].normal_theta(vec[i]);
            FLOAT current_theta = vec[i].normal_theta(vec[(i + 1) % elements_size]);
            if (fabs(prev_theta - current_theta) <= threshold) {
                remove(i);
                i--;
            }
        }
        return TRUE;
    }

   //returns state of the mask
    INT32 model::get_state(INT32*hist){
        if(area()>=1600&&!is_intersection()){
            //calculate the forces using the histogram
            forces(hist) ;
            //check for deformation
            for (UINT32 i = 0; i < elements_size; i++)
            {
                //if forces aren't zeros then deform
                if(std::fabs(force_vec[2*i])>force_threshold||std::fabs(force_vec[2*i+1])>force_threshold){
                    return DEFORM ;
                }
            }
            //now forces are stable

            //if error regions are bigger than limit then we need insert
            for(UINT32 i = 0 ; i<elements_size;i++){
                if ((hist[8 * i + A + 1] > (limit * vec[i].length(vec[(i + 1) % elements_size]) / 2)) || (hist[8 * i + B + 1] > (limit * vec[i].length(vec[(i + 1) % elements_size]) / 2)) ||
                    (hist[8 * i + C + 0] > (limit * vec[i].length(vec[(i + 1) % elements_size]) / 2)) || (hist[8 * i + D + 0] > (limit * vec[i].length(vec[(i + 1) % elements_size]) / 2))) {
                    return INSERT ;
                }
            }
            //else it's stable
            return STABLE ;
        }
        else {
            return REINIT ;
        }   
    }
void model::sort(void) {
    //finding most left and most right points
    UINT32 most_left_index = 0;
    UINT32 most_right_index = 0;
    for (UINT32 i = 0; i < elements_size; i++)
    {
        //assigning most left
        if (vec[i].x <= vec[most_left_index].x) {
            most_left_index = i;
        }
        //assigning most right
        if (vec[i].x > vec[most_right_index].x) {
            most_right_index = i;
        }
    }
    if (most_right_index == 0) {
        most_right_index = most_left_index;
    }
    swap(vec[0], vec[most_left_index]);
    swap(vec[1], vec[most_right_index]);
    most_right_index = 1;
    //y= ax+b
    //paritioning point into lower and upper planes
    FLOAT a = tan(vec[0].theta(vec[1]));
    FLOAT b = vec[0].y - a * vec[0].x;
    for (UINT32 i = 2; i < elements_size;i++) {
        //if upper plane shift most right index to the right
        //if its the element at most_right_index +1
        //then we are done
        //else we need to swap that element with the one that's is at upper plane
        if (vec[i].y > (a * vec[i].x + b)) {
            //shift most right element to the right
            swap(vec[most_right_index], vec[(most_right_index + 1)]);
            if (i != (most_right_index + 1)) {
                swap(vec[most_right_index], vec[i]);
            }
            most_right_index++;
        }
    }
    //sort upper plane ascendingly
    for (UINT32 i = 1; i < most_right_index - 1;i++) {
        for (UINT32 j = i + 1; j < most_right_index;j++) {
            if (vec[i].x == vec[j].x) {
                if (vec[i].y <= vec[j].y) {
                    swap(vec[i], vec[j]);
                }
            }
            else if (vec[i].x > vec[j].x) {
                swap(vec[i], vec[j]);
            }
        }
    }
    for (UINT32 i = most_right_index + 1; i < elements_size - 1;i++) {
        for (UINT32 j = i + 1; j < elements_size;j++) {
            if (vec[i].x == vec[j].x) {
                if (vec[i].y > vec[j].y) {
                    swap(vec[i], vec[j]);
                }
            }
            else if (vec[i].x <= vec[j].x) {
                swap(vec[i], vec[j]);
            }
        }
    }
}
//y=  ax+b 
BOOL line_eq(const point& p1,const point&p2,FLOAT &a,FLOAT&b){
    if(p1.x!=p2.x){
        a = static_cast<FLOAT>((p2.y-p1.y))/static_cast<FLOAT>((p2.x-p1.x) );
        b= -1*p1.x*a +p1.y ;
        return 1 ; 
    }
    //i.e., it's a perpendecular line (x=x1)
    return 0 ; 
}
void print_point(const point&p ){
    cout<<"\n("<<p.x<<","<<p.y<<") " ; 
}
BOOL model:: intersection_check(const point&p0,const point&p1, const point&p2 ,const point&p3) const{
    FLOAT a1=0,a2=0,b1=0,b2=0 ; 
    BOOL not_perp1 = line_eq(p0,p1,a1,b1) ;
    BOOL not_perp2 = line_eq(p2,p3,a2,b2) ;      
    INT32 len1 = p0.length_squared(p1) ;
    INT32 len2 = p2.length_squared(p3) ;  
    
    point inter_p ; 
    if(!not_perp1&&!not_perp2){
        //if both are pendecular then no chance of intersection
        return 0 ; 
    }
    if(not_perp1&&not_perp2){    
        if(fabs(a1-a2)>zero_threshold){
            //if less then they intersect at infinity 
            inter_p.x= round(static_cast<FLOAT>(b2-b1)/static_cast<FLOAT>(a1-a2)) ;    
            inter_p.y = round(a1*static_cast<FLOAT>(inter_p.x)+b1) ;
        }
        else{
             return 0;
        }
    }
    else if(not_perp2){
        //if first is perpendecular 
        inter_p.x = p0.x ; 
        inter_p.y = round(a2*static_cast<FLOAT>(inter_p.x)+b2) ;    
    }
    else if(not_perp1){
        //if second is perpendecular 
        inter_p.x = p2.x ; 
        inter_p.y =round(a1*static_cast<FLOAT>(inter_p.x)+b1 );    
    } 
    //if inter_p within first line and within second line
    if((inter_p.x>=0&&inter_p.x<=COLS-1 &&inter_p.y>=0&&inter_p.y<=ROWS-1&&inter_p.length_squared(p0)<len1&&inter_p.length_squared(p1)<len1)
     &&(inter_p.length_squared(p2)<len2&&inter_p.length_squared(p3)<len2)){
        return 1;
    }
    return 0; 
}   
BOOL model::is_intersection(void)const{
    for(UINT32 i = 0 ; i <elements_size -1; i++){
        for(UINT32 j=  i+1 ;j<elements_size;j++){
            if(intersection_check(vec[i],vec[(i+1)%elements_size],vec[j],vec[(j+1)%elements_size])){
              
                return 1; 
            }
        }
    }
    return 0 ; 
}
